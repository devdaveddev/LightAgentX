export const ease = [0.22, 1, 0.36, 1] as const;

export const REPO = "https://github.com/devdaveddev/LightAgentX";
export const DOCS = `${REPO}/blob/main/docs`;
export const SRC = `${REPO}/tree/main/lightagentx`;
export const PYPI = "https://pypi.org/project/lightagentx/";

// PyPI still has 0.1.0, which predates the sandbox, registry and doctor.
// Once 0.2.0 is uploaded, set ON_PYPI = true and the site switches to `pip install lightagentx`.
const ON_PYPI = false;
const GIT = "git+https://github.com/devdaveddev/LightAgentX";
export const install = (extra = "") =>
  ON_PYPI ? `pip install ${extra ? `"lightagentx[${extra}]"` : "lightagentx"}` : `pip install "lightagentx${extra ? `[${extra}]` : ""} @ ${GIT}"`;
