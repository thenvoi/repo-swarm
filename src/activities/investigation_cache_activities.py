"""
Activities for managing investigation caching.

These activities handle checking if a repository needs investigation
and saving investigation metadata for future checks.

Supports both DynamoDB (production) and file-based (local/CI) storage backends
based on the PROMPT_CONTEXT_STORAGE environment variable.
"""

import os
import logging
from typing import Dict, Optional
from datetime import datetime, timezone

from temporalio import activity

from .investigation_cache import InvestigationCache
from models.investigation import RepositoryState
from models.activities import CacheCheckInput, CacheCheckOutput, SaveMetadataInput, SaveMetadataOutput

# Set up logging
logger = logging.getLogger(__name__)


def _get_storage_client(repo_name: str):
    """
    Get the appropriate storage client based on PROMPT_CONTEXT_STORAGE env var.

    Args:
        repo_name: Repository name (needed for file-based storage initialization)

    Returns:
        Storage client instance (either DynamoDB or file-based)
    """
    storage_backend = os.environ.get('PROMPT_CONTEXT_STORAGE', 'auto')

    if storage_backend == 'file':
        logger.info(f"Using file-based storage backend for {repo_name}")
        from utils.prompt_context_file import FileBasedPromptContextManager
        return FileBasedPromptContextManager(repo_name)
    else:
        logger.info(f"Using DynamoDB storage backend for {repo_name}")
        from utils.dynamodb_client import get_dynamodb_client
        return get_dynamodb_client()


@activity.defn
async def check_if_repo_needs_investigation(
    input_params: CacheCheckInput
) -> CacheCheckOutput:
    """
    Check if a repository needs investigation based on the latest commit and previous investigations.
    
    Args:
        input_params: CacheCheckInput with repository details and prompt versions
    
    Returns:
        CacheCheckOutput with investigation decision details
    """
    activity.logger.info(f"🎯 ACTIVITY: check_if_repo_needs_investigation called for {input_params.repo_name}")
    activity.logger.info(f"   repo_url: {input_params.repo_url}")
    activity.logger.info(f"   repo_path: {input_params.repo_path}")
    if input_params.prompt_versions:
        activity.logger.info(f"   prompt_versions: {len(input_params.prompt_versions)} prompts provided")
        for name, version in input_params.prompt_versions.items():
            activity.logger.debug(f"      - {name}: v{version}")
    else:
        activity.logger.warning(f"   ⚠️  NO PROMPT VERSIONS provided to activity")
    
    try:
        # Get current repository state
        activity.logger.info(f"📊 Getting current repository state from {input_params.repo_path}")
        current_state = RepositoryState(
            commit_sha=_get_latest_commit(input_params.repo_path),
            branch_name=_get_current_branch(input_params.repo_path),
            has_uncommitted_changes=_has_uncommitted_changes(input_params.repo_path)
        )
        
        activity.logger.info(f"📊 Repository state: commit={current_state.commit_sha[:8]}, branch={current_state.branch_name}, uncommitted={current_state.has_uncommitted_changes}")

        # Get storage client based on environment configuration
        activity.logger.info(f"🗃️  Creating storage client and cache instance")
        storage_client = _get_storage_client(input_params.repo_name)
        cache = InvestigationCache(storage_client)
        
        # Check if investigation is needed
        activity.logger.info(f"🔍 Calling cache.check_needs_investigation...")
        decision = cache.check_needs_investigation(input_params.repo_name, current_state, input_params.prompt_versions)
        
        # Convert decision to CacheCheckOutput model
        activity.logger.info(f"✅ ACTIVITY RESULT: needs_investigation={decision.needs_investigation}, reason='{decision.reason}'")
        return CacheCheckOutput(
            needs_investigation=decision.needs_investigation,
            reason=decision.reason,
            latest_commit=decision.latest_commit,
            branch_name=decision.branch_name,
            last_investigation=decision.last_investigation
        )
        
    except Exception as e:
        activity.logger.error(f"💥 ACTIVITY ERROR: Error checking if repo needs investigation: {e}")
        # On error, default to investigating
        activity.logger.info(f"⚠️  ACTIVITY FALLBACK: Defaulting to needs_investigation=True due to error")
        
        # Try to still get commit info from the repository
        try:
            latest_commit = _get_latest_commit(input_params.repo_path)
            branch_name = _get_current_branch(input_params.repo_path)
            activity.logger.info(f"📊 Retrieved commit info despite error: commit={latest_commit[:8]}, branch={branch_name}")
        except Exception as git_error:
            activity.logger.warning(f"Could not retrieve git info: {git_error}")
            latest_commit = None
            branch_name = None
        
        return CacheCheckOutput(
            needs_investigation=True,
            reason=f"Unable to check previous investigations (storage error: {str(e)})",
            latest_commit=latest_commit,
            branch_name=branch_name,
            last_investigation=None
        )


@activity.defn
async def save_investigation_metadata(
    input_params: SaveMetadataInput
) -> SaveMetadataOutput:
    """
    Save investigation metadata for future caching checks.

    Uses the storage backend configured via PROMPT_CONTEXT_STORAGE env var
    (DynamoDB for production, file-based for local/CI).

    Args:
        input_params: SaveMetadataInput with repository details and analysis data

    Returns:
        SaveMetadataOutput with save status
    """
    activity.logger.info(f"🎯 ACTIVITY: save_investigation_metadata called for {input_params.repo_name}")
    activity.logger.info(f"   repo_url: {input_params.repo_url}")
    activity.logger.info(f"   latest_commit: {input_params.latest_commit[:8] if input_params.latest_commit else 'None'}")
    activity.logger.info(f"   branch_name: {input_params.branch_name}")
    activity.logger.info(f"   analysis_summary provided: {input_params.analysis_summary is not None}")
    if input_params.prompt_versions:
        activity.logger.info(f"   prompt_versions: {len(input_params.prompt_versions)} prompts provided")
        for name, version in input_params.prompt_versions.items():
            activity.logger.debug(f"      - {name}: v{version}")
    else:
        activity.logger.warning(f"   ⚠️  NO PROMPT VERSIONS provided to save metadata")
    
    try:
        # Get storage client based on environment configuration
        activity.logger.info(f"🗃️  Creating storage client and cache instance")
        storage_client = _get_storage_client(input_params.repo_name)
        cache = InvestigationCache(storage_client)
        
        # Save the investigation metadata
        activity.logger.info(f"💾 Calling cache.save_investigation_metadata...")
        result = cache.save_investigation_metadata(
            repo_name=input_params.repo_name,
            repo_url=input_params.repo_url,
            commit_sha=input_params.latest_commit,
            branch_name=input_params.branch_name,
            analysis_summary=input_params.analysis_summary,
            prompt_versions=input_params.prompt_versions,
            ttl_days=input_params.ttl_days
        )
        
        activity.logger.info(f"✅ ACTIVITY RESULT: save_investigation_metadata status={result.get('status')}")
        # Convert to SaveMetadataOutput model
        return SaveMetadataOutput(
            status=result.get('status', 'error'),
            message=result.get('message', ''),
            timestamp=result.get('timestamp')
        )
        
    except Exception as e:
        activity.logger.error(f"💥 ACTIVITY ERROR: Failed to save investigation metadata: {e}")
        return SaveMetadataOutput(
            status="error",
            message=f"Failed to save investigation metadata: {str(e)}",
            timestamp=None
        )


def _get_latest_commit(repo_path: str) -> str:
    """Get the latest commit SHA from the repository using GitPython."""
    try:
        import git
        repo = git.Repo(repo_path)
        # Get the HEAD commit SHA
        return repo.head.commit.hexsha
    except Exception as e:
        logger.error(f"Failed to get latest commit: {e}")
        raise


def _get_current_branch(repo_path: str) -> str:
    """Get the current branch name from the repository using GitPython."""
    try:
        import git
        repo = git.Repo(repo_path)
        # Get the active branch name
        # Handle detached HEAD state
        if repo.head.is_detached:
            return f"detached-{repo.head.commit.hexsha[:8]}"
        return repo.active_branch.name
    except Exception as e:
        logger.error(f"Failed to get current branch: {e}")
        raise


def _has_uncommitted_changes(repo_path: str) -> bool:
    """Check if the repository has uncommitted changes using GitPython."""
    try:
        import git
        repo = git.Repo(repo_path)
        # Check if there are any changes (staged or unstaged)
        return repo.is_dirty(untracked_files=True)
    except Exception as e:
        logger.error(f"Failed to check for uncommitted changes: {e}")
        # If we can't check, assume there might be changes
        return True
