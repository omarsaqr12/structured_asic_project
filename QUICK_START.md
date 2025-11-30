# Quick Start: X-Tree CTS GitHub Workflow

## Current Branch
✅ You are on: `feature/cts-xtree`

## Next Steps

### 1. Create GitHub Issue
Go to GitHub and create an issue with:
- **Title:** "Implement X-Tree Clock Tree Synthesis Algorithm"
- **Description:** Use the template at `.github/ISSUE_TEMPLATE/cts-xtree-implementation.md`
- **Labels:** `enhancement`, `cts`, `algorithm`
- **Assign to:** Yourself

### 2. Update Commit with Issue Number
After creating the issue, replace `<ISSUE_NUMBER>` with the actual number:

```bash
git commit --amend -m "Add X-Tree CTS algorithm with diagonal partitioning (fixes #<ISSUE_NUMBER>)"
```

### 3. Push Branch
```bash
git push origin feature/cts-xtree
```

### 4. Create Pull Request
On GitHub:
- Base: `main` ← Compare: `feature/cts-xtree`
- Title: "Implement X-Tree Clock Tree Synthesis Algorithm"
- Description: Use template from `.github/pull_request_template.md`
- Add: `Closes #<ISSUE_NUMBER>` in description
- Request review from team member

### 5. (Optional) Run X-Tree Now
You can run X-Tree in parallel while waiting for review:

```bash
python run_6502_cts_xtree.py
```

Results will be in:
- `build/6502/cts/greedy_xtree/`
- `build/6502/cts/best_xtree/`

---

## Files Ready for PR

✅ `cts_xtree.py` - X-Tree implementation (committed)
✅ `run_6502_cts_xtree.py` - Runner script (committed)
✅ Templates created for issue and PR

---

## Summary

**Branch:** `feature/cts-xtree`  
**Commit:** `a47e3d6` - "Add X-Tree CTS algorithm with diagonal partitioning (fixes #<issue_number>)"  
**Status:** Ready to push and create PR (after updating issue number)

