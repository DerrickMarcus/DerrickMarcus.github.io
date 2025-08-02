window.MathJax = {
    tex: {
        inlineMath: [["$", "$"], ["\\(", "\\)"]],
        displayMath: [["$$", "$$"], ["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true,
        packages: {
            "[+]": [
                /* 字体 / 符号 */
                "boldsymbol",   // \boldsymbol{}
                "upgreek",      // 直立希腊字母 \upalpha
                "gensymb",      // \degree, \ohm …

                /* 结构布局 */
                "mathtools",    // \prescript, \coloneqq 等增强
                "cases",        // 更灵活的分段公式
                "bbox",         // \bbox[5px,border:…]{…} 高亮/框
                "cancel",       // \cancel, \bcancel, \xcancel
                "enclose",      // \enclose{circle}{x}
                "extpfeil",     // \xRightarrow[下标]{上标}

                /* 高阶数学 / 物理 */
                "physics",      // \dv, \pdv, \qty, \commutator …
                "braket",       // \bra{}, \ket{}, \braket{}
                "bussproofs",   // 自然演绎推理树

                /* 颜色 */
                "color",        // \color{red}{…}
                "colorv2",      // 新版 xcolor 语法（grad, shade…）

                /* 化学与单位 */
                "mhchem",       // \ce{H2O}, \pu{3.5 m/s}
                "textmacros",   // \text{}, \textsf{} …（文字宏补全）

                /* 其它 */
                "unicode",      // 直接输入 Unicode 数学符号
                "verb"          // \verb|code| 行内等宽
            ]
        }
    },
    loader: {
        load: [/* 字体 / 符号 */
            "[tex]/boldsymbol", "[tex]/upgreek", "[tex]/gensymb",

            /* 结构布局 */
            "[tex]/mathtools", "[tex]/cases", "[tex]/bbox",
            "[tex]/cancel", "[tex]/enclose", "[tex]/extpfeil",

            /* 高阶数学 / 物理 */
            "[tex]/physics", "[tex]/braket", "[tex]/bussproofs",

            /* 颜色 */
            "[tex]/color", "[tex]/colorv2",

            /* 化学与单位 */
            "[tex]/mhchem", "[tex]/textmacros",

            /* 其它 */
            "[tex]/unicode", "[tex]/verb"
        ]
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    }
};

document$.subscribe(() => {
    MathJax.startup.output.clearCache()
    MathJax.typesetClear()
    MathJax.texReset()
    MathJax.typesetPromise()
})