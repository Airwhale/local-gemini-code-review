# GitLab MR source

Status: accepted. Date: 2026-10-13.

GitLab merge request IIDs are only meaningful within a project. Coding agents
must specify `--mr IID --repo https://HOST/group/project`; inference from a
checkout or glab defaults is forbidden. Each read-only glab call receives the
explicit full HTTPS repository URL. Public and self-hosted GitLab instances are
supported without embedding organization-specific defaults in the package.
Credential-bearing URLs, query strings, fragments, and non-HTTPS schemes are
rejected. Host selection comes only from the operator's CLI input.

Metadata uses a Pydantic v2 boundary model for `iid`, `web_url`, `sha`, and
optional `diff_refs`. Unknown API fields are ignored. The URL must match the
requested repository and IID exactly. `sha` is the head authority because older
MRs can have missing, empty, or null `diff_refs.head_sha`; populated diff refs must agree. Metadata
is read before and after the raw patch to reject concurrent head changes.

The exact validated URL is announced before model calls. The fetched patch and
validated metadata are reused within the request, avoiding separate file-list
fetches that could describe a newer revision. Destination path parsing supports
C-quoted UTF-8, spaces, rename-only changes, and deletions. glab output is decoded
from bytes without universal newline translation, and patch records are split
only at LF. Control characters and Unicode separators inside hunk bodies cannot
manufacture reference-file headers. Unquoted binary headers must contain an unambiguous pair
of paths or matching source/destination paths. Rename/copy metadata resolves
ambiguous headers; absent destination metadata is an error, never a guessed path.
The shared C-quoted UTF-8 decoder keeps hunk evidence aligned with quoted paths. Local context requires matching
HEAD and paths contained within the local repository. Dirty trees warn, matching
the existing GitHub policy. Automatic context falls back to hunks; explicit
full-file requests fail when context cannot be verified.

The existing source dispatch, provider layer, output schema, and exit codes are
retained. This source extension does not migrate the existing argparse CLI,
logging stack, or legacy internal records. All glab calls are read-only; account
configuration, comments, approvals, checkouts, and merges remain operator actions.
