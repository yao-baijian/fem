# Documentation System

This directory contains the complete documentation for the FPGA Placement FEM package.

## Structure

- `index.md` - Landing page
- `installation.md` - Installation guide
- `quickstart.md` - Quick start tutorial
- `USER_GUIDE.md` - Comprehensive user guide
- `API_REFERENCE.md` - Complete API documentation
- `ALGORITHM.md` - Algorithm and mathematical formulation
- `CODE_EXPLAINED.md` - **Deep dive into code implementation**
- `DEVELOPER_GUIDE.md` - Development and contributing guide
- `MIGRATION_GUIDE.md` - Migration guide from master branch
- `stylesheets/extra.css` - Custom CSS styling
- `javascripts/mathjax.js` - MathJax configuration

## Building Documentation

### Local Preview

```bash
# Install dependencies
pip install mkdocs mkdocs-material mkdocs-minify-plugin pymdown-extensions

# Serve locally (auto-reloads on changes)
mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

### Build Static Site

```bash
# Build to site/ directory
mkdocs build

# Build and check for errors
mkdocs build --strict

# Clean build
mkdocs build --clean
```

### Deploy to GitHub Pages

```bash
# Deploy manually
mkdocs gh-deploy

# Or push to master/main branch - GitHub Actions will deploy automatically
git push origin master
```

## GitHub Actions Deployment

Documentation is automatically deployed via GitHub Actions when you push to `master` or `main` branch.

**Workflow file**: `.github/workflows/docs.yml`

The workflow:
1. Checks out the repository
2. Installs Python and dependencies
3. Builds documentation with `mkdocs build`
4. Deploys to `gh-pages` branch
5. GitHub Pages serves from `gh-pages` branch

### Enabling GitHub Pages

1. Go to repository **Settings** → **Pages**
2. Set **Source** to "Deploy from a branch"
3. Select branch: `gh-pages`
4. Select folder: `/ (root)`
5. Click **Save**

Documentation will be available at: `https://yao-baijian.github.io/fem/`

## Configuration

### mkdocs.yml

Main configuration file at project root. Key sections:

```yaml
site_name: FPGA Placement FEM Documentation
theme:
  name: material
  palette:
    - scheme: default      # Light mode
    - scheme: slate        # Dark mode
  features:
    - navigation.instant   # Fast navigation
    - navigation.tabs      # Top-level tabs
    - search.suggest       # Search suggestions
    - content.code.copy    # Copy code button

nav:
  - Home: index.md
  - Getting Started: ...
  - User Guide: ...
  - API Reference: ...
  - Algorithm: ...
```

### Custom Styling

**CSS**: `docs/stylesheets/extra.css`
- Grid cards for features
- Custom table styling
- Dark mode adjustments
- Responsive design

**JavaScript**: `docs/javascripts/mathjax.js`
- Math equation rendering
- LaTeX support

## Writing Documentation

### Markdown Extensions

Supported features:

````markdown
# Headers with auto-linking

**Bold** and *italic* text

`inline code` and:

```python
# Code blocks with syntax highlighting
def example():
    return "Hello"
```

!!! note "Admonitions"
    Info boxes for notes, warnings, tips, etc.

| Tables | Are   | Supported |
|--------|-------|-----------|
| With   | Nice  | Styling   |

- Lists
  - Nested
  - Items

1. Numbered
2. Lists

[Links](https://example.com)

![Images](path/to/image.png)

Math: $E = mc^2$ or $$\int f(x) dx$$
````

### Admonitions

```markdown
!!! note "Optional Title"
    Content here

!!! tip
    Helpful tips

!!! warning
    Important warnings

!!! danger
    Critical information
```

### Tabs

````markdown
=== "Tab 1"
    Content for tab 1

=== "Tab 2"
    Content for tab 2
````

### Code Annotations

```python
def example():
    result = compute()  # (1)!
    return result

1. Explanation of this line
```

## Documentation Guidelines

### Style Guide

1. **Clear and Concise**: Use simple language
2. **Code Examples**: Always include working examples
3. **Consistent Format**: Follow existing patterns
4. **Up-to-Date**: Keep in sync with code changes

### Code Examples

Always include:
- Imports needed
- Complete, runnable examples
- Expected output
- Comments explaining key steps

**Good Example**:
```python
import torch
from fem_placer import FpgaPlacer

# Load design
placer = FpgaPlacer(utilization_factor=0.4)
placer.init_placement('design.dcp', 'output.dcp')

# Result: design loaded successfully
```

**Bad Example**:
```python
# Incomplete, can't run
placer.init_placement(file)
```

### API Documentation Format

For each function/class:

1. **Signature**: Show full function signature
2. **Description**: Brief description (1-2 sentences)
3. **Parameters**: List all parameters with types and descriptions
4. **Returns**: Describe return value(s)
5. **Example**: Working code example
6. **Notes**: Any important notes (optional)

## Updating Documentation

### When to Update

Update docs when you:
- Add new functions/classes
- Change API signatures
- Fix bugs that affect usage
- Add new features
- Improve performance significantly

### What to Update

1. **API Reference**: Add/update function documentation
2. **User Guide**: Add usage examples
3. **Code Explained**: Explain implementation
4. **Changelog**: Record changes

### Review Checklist

Before committing documentation changes:

- [ ] All code examples tested and working
- [ ] Links are valid
- [ ] Spelling and grammar checked
- [ ] Builds without errors (`mkdocs build --strict`)
- [ ] Previewed locally (`mkdocs serve`)
- [ ] Consistent with existing style

## Troubleshooting

### Build Errors

**Error: "Config file 'mkdocs.yml' does not exist"**
- Solution: Run from project root directory

**Error: "Theme 'material' not found"**
- Solution: `pip install mkdocs-material`

**Error: "Module 'pymdownx' not found"**
- Solution: `pip install pymdown-extensions`

### Broken Links

Check for broken links:
```bash
# Build with strict mode (fails on warnings)
mkdocs build --strict
```

### Math Not Rendering

Ensure MathJax is configured:
1. Check `docs/javascripts/mathjax.js` exists
2. Verify it's included in `mkdocs.yml` under `extra_javascript`
3. Use `$...$` for inline math, `$$...$$` for display math

### GitHub Pages Not Updating

1. Check GitHub Actions workflow passed
2. Verify `gh-pages` branch exists
3. Check repository Settings → Pages configuration
4. Clear browser cache
5. Wait 5-10 minutes for propagation

## Resources

- **MkDocs**: https://www.mkdocs.org/
- **Material Theme**: https://squidfunk.github.io/mkdocs-material/
- **Markdown Guide**: https://www.markdownguide.org/
- **PyMdown Extensions**: https://facelessuser.github.io/pymdown-extensions/
- **GitHub Pages**: https://pages.github.com/

## Questions?

For documentation-related questions:
- Check existing documentation
- Review [Material theme docs](https://squidfunk.github.io/mkdocs-material/)
- Open an issue on GitHub
