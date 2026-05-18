"""
Flow Tools - GitHub CI/CD Tools for Agents
===========================================
Lightweight HTTP stubs that call AitherFlow service (port 8165)
for GitHub Actions, CI/CD, releases, and code review.

These tools give agents the power to:
- Trigger CI/CD pipelines
- Create and merge PRs
- Run AI code reviews
- Manage releases
- Handle issues
- Configure branch protection
- Analyze GitHub Actions performance
- Build and modify workflows

All operations go through AitherFlow which handles GitHub authentication.
"""

import json
from typing import List

import httpx

# FLOW_URL from services.yaml


try:


    from lib.core.AitherPorts import get_service_url


    FLOW_URL = get_service_url("AitherFlow")


except ImportError:


    from lib.core.AitherPorts import get_port


    FLOW_URL = f"http://localhost:{get_port('AitherFlow', 8165)}"

# Reusable HTTP client
_flow_client: httpx.Client = None


def _get_flow_client() -> httpx.Client:
    """Get or create Flow HTTP client."""
    global _flow_client
    if _flow_client is None:
        _flow_client = httpx.Client(timeout=30.0)
    return _flow_client


def _flow_request(method: str, endpoint: str, data: dict = None, params: dict = None) -> str:
    """Make a request to AitherFlow service."""
    try:
        client = _get_flow_client()
        url = f"{FLOW_URL}{endpoint}"

        if method == "GET":
            response = client.get(url, params=params)
        elif method == "POST":
            response = client.post(url, json=data)
        elif method == "PUT":
            response = client.put(url, json=data)
        elif method == "PATCH":
            response = client.patch(url, json=data)
        elif method == "DELETE":
            response = client.delete(url)
        else:
            return json.dumps({"error": f"Unknown method: {method}"})

        if response.status_code >= 400:
            return json.dumps({"error": f"HTTP {response.status_code}: {response.text}"})

        try:
            return json.dumps(response.json(), indent=2)
        except ValueError:
            return response.text

    except httpx.ConnectError:
        return json.dumps({"error": "Cannot connect to AitherFlow. Is it running on port 8165?"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# WORKFLOW TOOLS
# =============================================================================

def github_list_workflows() -> str:
    """
    List all GitHub Actions workflows in the repository.

    Use this to see what CI/CD pipelines are available.

    Returns:
        JSON list of workflows with id, name, state, and path.
    """
    return _flow_request("GET", "/workflows")


def github_trigger_workflow(workflow_id: str, ref: str = "main", inputs: dict = None) -> str:
    """
    Trigger a GitHub Actions workflow.

    Use this to start a CI pipeline, run tests, or deploy.

    Args:
        workflow_id: The workflow ID or filename (e.g., "ci.yml")
        ref: Branch or tag to run on (default: "main")
        inputs: Optional workflow inputs dict

    Returns:
        JSON with trigger result

    Example:
        github_trigger_workflow("ci.yml", "feature-branch")
    """
    return _flow_request("POST", f"/workflows/{workflow_id}/dispatch", {
        "ref": ref,
        "inputs": inputs or {}
    })


def github_get_workflow_runs(workflow_id: str, limit: int = 10) -> str:
    """
    Get recent runs of a workflow.

    Use this to check pipeline history and results.

    Args:
        workflow_id: The workflow ID or filename
        limit: Max number of runs to return (default: 10)

    Returns:
        JSON list of workflow runs with status and conclusion
    """
    return _flow_request("GET", f"/workflows/{workflow_id}/runs", params={"limit": limit})


def github_cancel_workflow(run_id: int) -> str:
    """
    Cancel a running workflow.

    Args:
        run_id: The workflow run ID to cancel

    Returns:
        JSON with cancellation result
    """
    return _flow_request("POST", f"/workflows/runs/{run_id}/cancel")


def github_rerun_workflow(run_id: int) -> str:
    """
    Rerun a failed workflow.

    Args:
        run_id: The workflow run ID to rerun

    Returns:
        JSON with rerun result
    """
    return _flow_request("POST", f"/workflows/runs/{run_id}/rerun")


# =============================================================================
# CI/CD TOOLS
# =============================================================================

def github_run_ci_tests(branch: str = "main") -> str:
    """
    Run CI tests on a branch.

    This triggers the ci.yml workflow with test suite.
    Use after making changes to verify they work.

    Args:
        branch: Branch to test (default: "main")

    Returns:
        JSON with CI run details
    """
    return _flow_request("POST", "/ci/test", {"branch": branch})


def github_run_security_scan(branch: str = "main") -> str:
    """
    Run security vulnerability scan on a branch.

    Triggers security-scan.yml workflow with SAST analysis.
    Use before deploying to production.

    Args:
        branch: Branch to scan

    Returns:
        JSON with security scan details
    """
    return _flow_request("POST", "/ci/security", {"branch": branch})


def github_ci_status() -> str:
    """
    Get overall CI/CD status for the repository.

    Shows recent workflow runs, pass/fail rates, and any active issues.

    Returns:
        JSON with CI status summary
    """
    return _flow_request("GET", "/ci/status")


# =============================================================================
# PR TOOLS
# =============================================================================

def github_list_prs(state: str = "open") -> str:
    """
    List pull requests in the repository.

    Args:
        state: "open", "closed", or "all" (default: "open")

    Returns:
        JSON list of PRs with title, author, status
    """
    return _flow_request("GET", "/prs", params={"state": state})


def github_ai_review_pr(pr_number: int) -> str:
    """
    Run AI code review on a pull request.

    This analyzes the PR diff and provides:
    - Code quality feedback
    - Bug detection
    - Security concerns
    - Suggested improvements

    Args:
        pr_number: The PR number to review

    Returns:
        JSON with AI review results
    """
    return _flow_request("POST", f"/prs/{pr_number}/review")


def github_merge_pr(pr_number: int, merge_method: str = "squash") -> str:
    """
    Merge a pull request.

    Use after review approval. Supports:
    - squash: Combine all commits (default, cleaner history)
    - merge: Create merge commit (preserves history)
    - rebase: Rebase onto base branch (linear history)

    Args:
        pr_number: The PR number to merge
        merge_method: "squash", "merge", or "rebase"

    Returns:
        JSON with merge result
    """
    return _flow_request("POST", f"/prs/{pr_number}/merge", {"merge_method": merge_method})


def github_comment_on_pr(pr_number: int, comment: str) -> str:
    """
    Add a comment to a pull request.

    Use for feedback, questions, or documenting decisions.
    Supports Markdown formatting.

    Args:
        pr_number: The PR number
        comment: Comment text (Markdown supported)

    Returns:
        JSON with comment result
    """
    return _flow_request("POST", f"/prs/{pr_number}/comments", {"body": comment})


def github_get_pr_diff(pr_number: int) -> str:
    """
    Get the diff for a pull request.

    Returns unified diff showing all code changes.
    Use to understand what a PR modifies.

    Args:
        pr_number: The PR number

    Returns:
        The diff text
    """
    return _flow_request("GET", f"/prs/{pr_number}/diff")


def github_get_pr_files(pr_number: int) -> str:
    """
    Get list of files changed in a PR.

    Shows file paths, change type (added/modified/removed),
    and line statistics.

    Args:
        pr_number: The PR number

    Returns:
        JSON list of changed files
    """
    return _flow_request("GET", f"/prs/{pr_number}/files")


def github_enable_auto_merge(pr_number: int, merge_method: str = "SQUASH") -> str:
    """
    Enable auto-merge for a PR.

    The PR will automatically merge when:
    - All required status checks pass
    - All required reviews are approved

    Args:
        pr_number: The PR number
        merge_method: "SQUASH", "MERGE", or "REBASE"

    Returns:
        JSON with auto-merge status
    """
    return _flow_request("POST", f"/prs/{pr_number}/auto-merge", {"merge_method": merge_method})


# =============================================================================
# ISSUE TOOLS
# =============================================================================

def github_list_issues(state: str = "open", labels: str = None) -> str:
    """
    List GitHub issues in the repository.

    Args:
        state: "open", "closed", or "all"
        labels: Comma-separated label filter

    Returns:
        JSON list of issues
    """
    params = {"state": state, "source": "github"}
    if labels:
        params["labels"] = labels
    return _flow_request("GET", "/issues", params=params)


def github_create_issue(title: str, body: str, labels: List[str] = None) -> str:
    """
    Create a new GitHub issue.

    Args:
        title: Issue title
        body: Issue description (Markdown supported)
        labels: Optional list of labels

    Returns:
        JSON with created issue details

    Example:
        github_create_issue(
            "Add dark mode support",
            "Users have requested a dark theme option.",
            ["enhancement", "ui"]
        )
    """
    return _flow_request("POST", "/issues", {
        "title": title,
        "body": body,
        "labels": labels or []
    })


def github_assign_issue_to_agent(issue_number: int, agent_name: str) -> str:
    """
    Assign a GitHub issue to an AitherOS agent.

    The agent will be notified and can start working on it.

    Args:
        issue_number: The issue number
        agent_name: Agent to assign (e.g., "InfraAgent", "Saga")

    Returns:
        JSON with assignment result
    """
    return _flow_request("POST", f"/issues/{issue_number}/assign", {"agent": agent_name})


# =============================================================================
# RELEASE TOOLS
# =============================================================================

def github_create_release(tag: str, name: str = None, body: str = None, prerelease: bool = False) -> str:
    """
    Create a new GitHub release.

    Args:
        tag: Version tag (e.g., "v1.0.0")
        name: Release name (defaults to tag)
        body: Release notes (Markdown)
        prerelease: Mark as pre-release

    Returns:
        JSON with release details

    Example:
        github_create_release(
            "v1.2.0",
            "AitherOS 1.2.0",
            "## What's New\\n- Feature X\\n- Bug fix Y",
            False
        )
    """
    return _flow_request("POST", "/releases", {
        "tag": tag,
        "name": name or tag,
        "body": body or "",
        "prerelease": prerelease
    })


def github_list_releases(limit: int = 10) -> str:
    """
    List recent releases.

    Args:
        limit: Max releases to return

    Returns:
        JSON list of releases
    """
    return _flow_request("GET", "/releases", params={"limit": limit})


def github_get_release_by_tag(tag: str) -> str:
    """
    Get a specific release by its tag name.

    Args:
        tag: The release tag (e.g., "v2.1.0")

    Returns:
        JSON with release details including name, body, assets, draft/prerelease status
    """
    return _flow_request("GET", f"/releases/tag/{tag}")


def github_update_release(release_id: int, name: str = None, body: str = None, draft: bool = None, prerelease: bool = None) -> str:
    """
    Update an existing release. Use to mark as draft, change notes, etc.

    Args:
        release_id: The release ID (from list_releases)
        name: New name (optional)
        body: New release notes (optional)
        draft: Set to True to hide release, False to publish (optional)
        prerelease: Set prerelease flag (optional)

    Returns:
        JSON with updated release
    """
    data = {}
    if name is not None:
        data["name"] = name
    if body is not None:
        data["body"] = body
    if draft is not None:
        data["draft"] = draft
    if prerelease is not None:
        data["prerelease"] = prerelease
    return _flow_request("PATCH", f"/releases/{release_id}", data)


def github_delete_release(release_id: int) -> str:
    """
    Permanently delete a release and its assets.

    ⚠️ This is destructive! Consider disqualify instead.

    Args:
        release_id: The release ID to delete

    Returns:
        JSON confirmation
    """
    return _flow_request("DELETE", f"/releases/{release_id}")


def github_list_tags(limit: int = 30) -> str:
    """
    List git tags in the repository.

    Args:
        limit: Max tags to return (default: 30)

    Returns:
        JSON list of tags with name and commit sha
    """
    return _flow_request("GET", "/tags", params={"limit": limit})


# =============================================================================
# RELEASE MANAGER TOOLS (Versioning & Rollback Workflows)
# =============================================================================

def github_trigger_release(version: str, release_type: str = "stable", skip_tests: bool = False, dry_run: bool = False, release_notes: str = "") -> str:
    """
    Trigger the Release Manager workflow to create a new versioned release.

    This runs the full release pipeline:
    1. Validates the version (semver format)
    2. Runs tests (unless skipped)
    3. Bumps VERSION file + AitherZero.psd1
    4. Creates git tag
    5. Builds artifacts
    6. Publishes GitHub Release

    Args:
        version: Semver version string (e.g., "2.1.0" or "2.1.0-beta.1")
        release_type: One of "stable", "beta", "rc", "hotfix" (default: "stable")
        skip_tests: Skip validation tests (emergency only, default: False)
        dry_run: Validate only without publishing (default: False)
        release_notes: Custom release notes (auto-generated if empty)

    Returns:
        JSON with workflow trigger confirmation

    Example:
        github_trigger_release("2.1.0", "stable")
        github_trigger_release("2.2.0-beta.1", "beta", dry_run=True)
    """
    return _flow_request("POST", "/release-manager/create", {
        "version": version,
        "release_type": release_type,
        "skip_tests": skip_tests,
        "dry_run": dry_run,
        "release_notes": release_notes
    })


def github_rollback_release(tag: str, action: str, reason: str, rollback_to: str = "", delete_tag: bool = False) -> str:
    """
    Rollback, disqualify, cleanup, or restore a release.

    Actions:
    - "rollback": Revert VERSION files to previous release, mark as draft
    - "disqualify": Flag as bad build (kept for audit, marked DQ)
    - "cleanup": Permanently delete release + tag (creates audit issue)
    - "restore": Un-DQ a previously disqualified release

    Args:
        tag: The release tag to act on (e.g., "v2.1.0")
        action: One of "rollback", "disqualify", "cleanup", "restore"
        reason: Why (required for audit trail)
        rollback_to: Specific tag to rollback to (optional, defaults to previous)
        delete_tag: Also delete the git tag (for rollback, default: False)

    Returns:
        JSON with workflow trigger confirmation

    Example:
        github_rollback_release("v2.1.0", "disqualify", "Broke auth endpoints")
        github_rollback_release("v2.1.0", "rollback", "Critical bug", "v2.0.0")
    """
    return _flow_request("POST", "/release-manager/rollback", {
        "tag": tag,
        "action": action,
        "reason": reason,
        "rollback_to": rollback_to,
        "delete_tag": delete_tag
    })


def github_release_status() -> str:
    """
    Get current release management status.

    Shows:
    - Current version
    - Recent release workflow runs
    - Recent rollback workflow runs
    - Latest releases and tags

    Returns:
        JSON with comprehensive release status
    """
    return _flow_request("GET", "/release-manager/status")


# =============================================================================
# LABELS TOOLS
# =============================================================================

def github_list_labels() -> str:
    """
    List all labels in the repository.

    Returns:
        JSON list of labels with name, color, description
    """
    return _flow_request("GET", "/labels")


def github_create_label(name: str, color: str, description: str = "") -> str:
    """
    Create a new label.

    Args:
        name: Label name (e.g., "bug", "enhancement")
        color: Hex color without # (e.g., "d73a4a")
        description: What this label means

    Returns:
        JSON with created label
    """
    return _flow_request("POST", "/labels", {
        "name": name,
        "color": color,
        "description": description
    })


def github_delete_label(name: str) -> str:
    """
    Delete a label.

    Args:
        name: Label name to delete

    Returns:
        JSON with result
    """
    return _flow_request("DELETE", f"/labels/{name}")


# =============================================================================
# MILESTONES TOOLS
# =============================================================================

def github_list_milestones(state: str = "open") -> str:
    """
    List milestones in the repository.

    Args:
        state: "open", "closed", or "all"

    Returns:
        JSON list of milestones with progress
    """
    return _flow_request("GET", "/milestones", params={"state": state})


def github_create_milestone(title: str, description: str = "", due_on: str = None) -> str:
    """
    Create a new milestone.

    Args:
        title: Milestone title (e.g., "v1.0 Release")
        description: What this milestone represents
        due_on: Due date in ISO format (YYYY-MM-DD)

    Returns:
        JSON with created milestone
    """
    data = {"title": title, "description": description}
    if due_on:
        data["due_on"] = due_on
    return _flow_request("POST", "/milestones", data)


def github_close_milestone(number: int) -> str:
    """
    Close a milestone.

    Args:
        number: Milestone number

    Returns:
        JSON with result
    """
    return _flow_request("PATCH", f"/milestones/{number}", {"state": "closed"})


# =============================================================================
# BRANCH PROTECTION TOOLS
# =============================================================================

def github_get_branch_protection(branch: str = "main") -> str:
    """
    Get branch protection rules.

    Args:
        branch: Branch name (default: "main")

    Returns:
        JSON with protection rules
    """
    return _flow_request("GET", f"/branches/{branch}/protection")


def github_update_branch_protection(
    branch: str = "main",
    required_reviews: int = 1,
    dismiss_stale_reviews: bool = True,
    require_code_owner_reviews: bool = False,
    required_status_checks: List[str] = None,
    strict: bool = True
) -> str:
    """
    Update branch protection rules.

    Use to enforce code quality and review requirements.

    Args:
        branch: Branch name
        required_reviews: Number of approvals needed
        dismiss_stale_reviews: Dismiss approvals on new commits
        require_code_owner_reviews: Require CODEOWNERS review
        required_status_checks: List of required CI checks
        strict: Require branches be up to date

    Returns:
        JSON with updated rules

    Example:
        github_update_branch_protection(
            "main",
            required_reviews=2,
            required_status_checks=["test", "lint"]
        )
    """
    return _flow_request("PUT", f"/branches/{branch}/protection", {
        "required_reviews": required_reviews,
        "dismiss_stale_reviews": dismiss_stale_reviews,
        "require_code_owner_reviews": require_code_owner_reviews,
        "required_status_checks": required_status_checks or [],
        "strict": strict
    })


# =============================================================================
# SECRETS TOOLS
# =============================================================================

def github_update_secret(name: str, value: str) -> str:
    """
    Update a repository secret.

    Secrets are encrypted and only available to workflows.

    Args:
        name: Secret name (e.g., "API_KEY")
        value: Secret value (will be encrypted)

    Returns:
        JSON with result
    """
    return _flow_request("PUT", f"/secrets/{name}", {"value": value})


# =============================================================================
# PROJECTS TOOLS
# =============================================================================

def github_list_projects() -> str:
    """
    List GitHub Projects (v2) for the repository.

    Returns:
        JSON list of projects
    """
    return _flow_request("GET", "/projects")


# =============================================================================
# ISSUE MANAGEMENT (EXPANDED)
# =============================================================================

def github_update_issue(issue_number: int, title: str = None, body: str = None, state: str = None, labels: List[str] = None) -> str:
    """
    Update an existing GitHub issue.

    Args:
        issue_number: Issue number to update
        title: New title (optional)
        body: New body (optional)
        state: New state - 'open' or 'closed' (optional)
        labels: Replace labels list (optional)

    Returns:
        JSON with updated issue
    """
    data = {}
    if title: data["title"] = title
    if body: data["body"] = body
    if state: data["state"] = state
    if labels is not None: data["labels"] = labels
    return _flow_request("PATCH", f"/issues/{issue_number}", data)


def github_search_issues(query: str, state: str = "open") -> str:
    """
    Search GitHub issues with a query string.

    Args:
        query: Search query (GitHub search syntax)
        state: Filter by state - 'open', 'closed', or 'all'

    Returns:
        JSON list of matching issues
    """
    return _flow_request("GET", "/issues/search", params={"q": query, "state": state})


def github_list_issue_comments(issue_number: int) -> str:
    """
    List comments on a GitHub issue.

    Args:
        issue_number: Issue number

    Returns:
        JSON list of comments
    """
    return _flow_request("GET", f"/issues/{issue_number}/comments")


def github_add_issue_comment(issue_number: int, body: str) -> str:
    """
    Add a comment to a GitHub issue.

    Args:
        issue_number: Issue number to comment on
        body: Comment text

    Returns:
        JSON with created comment
    """
    return _flow_request("POST", f"/issues/{issue_number}/comments", {"body": body})


# =============================================================================
# PR MANAGEMENT (EXPANDED)
# =============================================================================

def github_update_pr(pr_number: int, title: str = None, body: str = None, state: str = None) -> str:
    """
    Update an existing pull request.

    Args:
        pr_number: PR number to update
        title: New title (optional)
        body: New body (optional)
        state: New state - 'open' or 'closed' (optional)

    Returns:
        JSON with updated PR
    """
    data = {}
    if title: data["title"] = title
    if body: data["body"] = body
    if state: data["state"] = state
    return _flow_request("PATCH", f"/prs/{pr_number}", data)


def github_list_pr_reviews(pr_number: int) -> str:
    """
    List reviews on a pull request.

    Args:
        pr_number: PR number

    Returns:
        JSON list of reviews with state, user, body
    """
    return _flow_request("GET", f"/prs/{pr_number}/reviews")


def github_request_reviewers(pr_number: int, reviewers: List[str]) -> str:
    """
    Request reviewers for a pull request.

    Args:
        pr_number: PR number
        reviewers: List of GitHub usernames to request review from

    Returns:
        JSON confirmation
    """
    return _flow_request("POST", f"/prs/{pr_number}/reviewers", {"reviewers": reviewers})


def github_list_pr_commits(pr_number: int) -> str:
    """
    List commits in a pull request.

    Args:
        pr_number: PR number

    Returns:
        JSON list of commits
    """
    return _flow_request("GET", f"/prs/{pr_number}/commits")


# =============================================================================
# COMMIT TOOLS
# =============================================================================

def github_list_commits(sha: str = None, limit: int = 20) -> str:
    """
    List recent commits on the repository.

    Args:
        sha: Branch name or commit SHA to start from (optional)
        limit: Max commits to return (default: 20)

    Returns:
        JSON list of commits
    """
    params = {"per_page": limit}
    if sha: params["sha"] = sha
    return _flow_request("GET", "/commits", params=params)


def github_get_commit(sha: str) -> str:
    """
    Get details of a specific commit.

    Args:
        sha: Commit SHA

    Returns:
        JSON with commit details, files changed, stats
    """
    return _flow_request("GET", f"/commits/{sha}")


def github_compare_commits(base: str, head: str) -> str:
    """
    Compare two commits/branches/tags.

    Args:
        base: Base ref (branch, tag, or SHA)
        head: Head ref (branch, tag, or SHA)

    Returns:
        JSON with comparison including ahead_by, behind_by, commits, files
    """
    return _flow_request("GET", f"/compare/{base}...{head}")


# =============================================================================
# ACTIONS ANALYTICS
# =============================================================================

def github_actions_analytics(days: int = 30, workflow_id: int = None) -> str:
    """
    Get GitHub Actions analytics: success rates, duration trends, failure tracking.

    Args:
        days: Number of days to analyze (default: 30)
        workflow_id: Filter to a specific workflow (optional)

    Returns:
        JSON with total_runs, success_rate, avg_duration, per_workflow breakdown,
        recent_failures, and slowest_workflows
    """
    params = {"days": days}
    if workflow_id: params["workflow_id"] = workflow_id
    return _flow_request("GET", "/actions/analytics", params=params)


def github_list_workflow_jobs(run_id: int) -> str:
    """
    List jobs for a workflow run.

    Args:
        run_id: Workflow run ID

    Returns:
        JSON list of jobs with status, conclusion, steps
    """
    return _flow_request("GET", f"/workflows/runs/{run_id}/jobs")


def github_delete_workflow_run(run_id: int) -> str:
    """
    Delete a workflow run.

    Args:
        run_id: Workflow run ID to delete

    Returns:
        JSON confirmation
    """
    return _flow_request("DELETE", f"/workflows/runs/{run_id}")


# =============================================================================
# ACTION BUILDER
# =============================================================================

def github_build_workflow(name: str, filename: str, triggers: dict, jobs: list, description: str = "", dry_run: bool = True) -> str:
    """
    Build a GitHub Actions workflow YAML from structured input.

    Args:
        name: Workflow name
        filename: YAML filename (e.g., "ci.yml")
        triggers: Trigger configuration dict (e.g., {"push": {"branches": ["main"]}})
        jobs: List of job dicts with name, runs_on, steps
        description: Optional description
        dry_run: If True, only preview YAML without writing (default: True)

    Returns:
        JSON with generated YAML, validation info
    """
    return _flow_request("POST", "/actions/build-workflow", {
        "name": name,
        "filename": filename,
        "description": description,
        "triggers": triggers,
        "jobs": jobs,
        "dry_run": dry_run
    })


def github_list_workflows_on_disk() -> str:
    """
    List all GitHub Actions workflow files in the repository.

    Returns:
        JSON list of workflow files with name, triggers, jobs_count
    """
    return _flow_request("GET", "/actions/workflows-on-disk")


def github_get_workflow_file(filename: str) -> str:
    """
    Read the content of a workflow file.

    Args:
        filename: Workflow filename (e.g., "ci.yml")

    Returns:
        JSON with filename, content, parsed metadata
    """
    return _flow_request("GET", f"/actions/workflow-file/{filename}")


def github_update_workflow_file(filename: str, content: str) -> str:
    """
    Update a workflow file with new content. Validates YAML and commits.

    Args:
        filename: Workflow filename to update
        content: New YAML content

    Returns:
        JSON confirmation with commit info
    """
    return _flow_request("PUT", f"/actions/workflow-file/{filename}", {"content": content})


# =============================================================================
# REPOSITORY INFO
# =============================================================================

def github_get_repo_info() -> str:
    """
    Get repository information including name, description, stars, forks, etc.

    Returns:
        JSON with full repository metadata
    """
    return _flow_request("GET", "/repo")


def github_get_rate_limit() -> str:
    """
    Get GitHub API rate limit status.

    Returns:
        JSON with remaining calls, reset time, used calls
    """
    return _flow_request("GET", "/rate-limit")


def github_delete_branch(branch: str) -> str:
    """
    Delete a branch. Protected branches (main, master, develop) cannot be deleted.

    Args:
        branch: Branch name to delete

    Returns:
        JSON confirmation
    """
    return _flow_request("DELETE", f"/branches/{branch}")


# =============================================================================
# TOOL LIST FOR REGISTRATION
# =============================================================================

flow_tools = [
    # Workflows
    github_list_workflows,
    github_trigger_workflow,
    github_get_workflow_runs,
    github_cancel_workflow,
    github_rerun_workflow,
    # CI/CD
    github_run_ci_tests,
    github_run_security_scan,
    github_ci_status,
    # PRs
    github_list_prs,
    github_ai_review_pr,
    github_merge_pr,
    github_comment_on_pr,
    github_get_pr_diff,
    github_get_pr_files,
    github_enable_auto_merge,
    github_update_pr,
    github_list_pr_reviews,
    github_request_reviewers,
    github_list_pr_commits,
    # Issues
    github_list_issues,
    github_create_issue,
    github_assign_issue_to_agent,
    github_update_issue,
    github_search_issues,
    github_list_issue_comments,
    github_add_issue_comment,
    # Releases
    github_create_release,
    github_list_releases,
    github_get_release_by_tag,
    github_update_release,
    github_delete_release,
    github_list_tags,
    github_trigger_release,
    github_rollback_release,
    github_release_status,
    # Labels
    github_list_labels,
    github_create_label,
    github_delete_label,
    # Milestones
    github_list_milestones,
    github_create_milestone,
    github_close_milestone,
    # Branch Protection
    github_get_branch_protection,
    github_update_branch_protection,
    # Secrets
    github_update_secret,
    # Projects
    github_list_projects,
    # Commits
    github_list_commits,
    github_get_commit,
    github_compare_commits,
    # Actions Analytics
    github_actions_analytics,
    github_list_workflow_jobs,
    github_delete_workflow_run,
    # Action Builder
    github_build_workflow,
    github_list_workflows_on_disk,
    github_get_workflow_file,
    github_update_workflow_file,
    # Repo Info
    github_get_repo_info,
    github_get_rate_limit,
    github_delete_branch,
]

__all__ = [
    "flow_tools",
    # Workflows
    "github_list_workflows",
    "github_trigger_workflow",
    "github_get_workflow_runs",
    "github_cancel_workflow",
    "github_rerun_workflow",
    # CI/CD
    "github_run_ci_tests",
    "github_run_security_scan",
    "github_ci_status",
    # PRs
    "github_list_prs",
    "github_ai_review_pr",
    "github_merge_pr",
    "github_comment_on_pr",
    "github_get_pr_diff",
    "github_get_pr_files",
    "github_enable_auto_merge",
    "github_update_pr",
    "github_list_pr_reviews",
    "github_request_reviewers",
    "github_list_pr_commits",
    # Issues
    "github_list_issues",
    "github_create_issue",
    "github_assign_issue_to_agent",
    "github_update_issue",
    "github_search_issues",
    "github_list_issue_comments",
    "github_add_issue_comment",
    # Releases
    "github_create_release",
    "github_list_releases",
    "github_get_release_by_tag",
    "github_update_release",
    "github_delete_release",
    "github_list_tags",
    "github_trigger_release",
    "github_rollback_release",
    "github_release_status",
    # Labels
    "github_list_labels",
    "github_create_label",
    "github_delete_label",
    # Milestones
    "github_list_milestones",
    "github_create_milestone",
    "github_close_milestone",
    # Branch Protection
    "github_get_branch_protection",
    "github_update_branch_protection",
    # Secrets
    "github_update_secret",
    # Projects
    "github_list_projects",
    # Commits
    "github_list_commits",
    "github_get_commit",
    "github_compare_commits",
    # Actions Analytics
    "github_actions_analytics",
    "github_list_workflow_jobs",
    "github_delete_workflow_run",
    # Action Builder
    "github_build_workflow",
    "github_list_workflows_on_disk",
    "github_get_workflow_file",
    "github_update_workflow_file",
    # Repo Info
    "github_get_repo_info",
    "github_get_rate_limit",
    "github_delete_branch",
]
