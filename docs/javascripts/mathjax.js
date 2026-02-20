// MathJax configuration for rendering mathematical equations

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['ams', 'color', 'bbox']}
  },
  svg: {
    fontCache: 'global'
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  loader: {
    load: ['[tex]/ams', '[tex]/color', '[tex]/bbox']
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
