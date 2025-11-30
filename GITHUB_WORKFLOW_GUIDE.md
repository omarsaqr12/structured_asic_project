# GitHub Workflow Guide for X-Tree CTS Implementation

## Step-by-Step Workflow

### 1. Create GitHub Issue
Create an issue on GitHub using the template at `.github/ISSUE_TEMPLATE/cts-xtree-implementation.md`:

**Issue Title:** `Implement X-Tree Clock Tree Synthesis Algorithm`

**Issue Body:** (Copy from the template or use the checklist below)
- [x] Create `cts_xtree.py` with diagonal partitioning logic
- [x] Implement alternating diagonal splits (NW/SE and NE/SW)
- [x] Use same buffer selection and tree structure as H-Tree
- [x] Create runner script `run_6502_cts_xtree.py`
- [ ] Run X-Tree on 6502 design
- [ ] Compare X-Tree vs H-Tree metrics

**Labels:** `enhancement`, `cts`, `algorithm`

**Assign to:** Yourself

---

### 2. Update Commit Message with Issue Number
After creating the issue, update the commit message:

```bash
git commit --amend -m "Add X-Tree CTS algorithm with diagonal partitioning (fixes #<ISSUE_NUMBER>)"
```

Replace `<ISSUE_NUMBER>` with the actual GitHub issue number.

---

### 3. Push Branch to Remote
```bash
git push origin feature/cts-xtree
```

---

### 4. Create Pull Request
On GitHub:
1. Go to the repository
2. Click "New Pull Request"
3. Select `feature/cts-xtree` → `main`
4. Use the PR template from `.github/pull_request_template.md`
5. **Link the issue** by adding `Closes #<ISSUE_NUMBER>` in the PR description
6. Request review from at least one team member

**PR Title:** `Implement X-Tree Clock Tree Synthesis Algorithm`

**PR Description:** (Use template or include)
- Description of changes
- Related issue number
- Testing checklist
- Comparison metrics (to be added)

---

### 5. Code Review Process
- Reviewer should:
  - Test the X-Tree implementation
  - Verify output doesn't override H-Tree results
  - Check code quality and consistency
  - Review diagonal partitioning logic
- Address any review comments
- Push additional commits if needed (they will auto-update the PR)

---

### 6. Merge to Main
Once approved:
- Merge the PR (squash merge recommended)
- The linked issue will automatically close
- Delete the feature branch after merge

---

## Current Status

✅ **Completed:**
- Created `cts_xtree.py` with X-Tree implementation
- Created `run_6502_cts_xtree.py` runner script
- Created feature branch: `feature/cts-xtree`
- Committed changes (needs issue number update)
- Created issue and PR templates

⏳ **Next Steps:**
1. Create GitHub issue and get issue number
2. Update commit message with issue number
3. Push branch: `git push origin feature/cts-xtree`
4. Create Pull Request on GitHub
5. Request code review
6. Run X-Tree on 6502 design (can be done in parallel)

---

## Files Changed

### New Files
- `cts_xtree.py` - X-Tree CTS implementation (800+ lines)
- `run_6502_cts_xtree.py` - Runner script for 6502 design
- `.github/ISSUE_TEMPLATE/cts-xtree-implementation.md` - Issue template
- `.github/pull_request_template.md` - PR template

### Output Directories (when run)
- `build/6502/cts/greedy_xtree/` - X-Tree results for greedy placement
- `build/6502/cts/best_xtree/` - X-Tree results for best SA placement

---

## Running X-Tree CTS

To run X-Tree on 6502 design:
```bash
python run_6502_cts_xtree.py
```

This will generate results in separate directories that don't override H-Tree results.

