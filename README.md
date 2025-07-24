# DerrickMarcus.github.io

Welcome!

This is the personal website of Yanxu Chen.

I'm currently an undergraduate student in the Department of Electronic Engineering, Tsinghua University.

In this site I plan to share some knowledge and skills during my study in computer, focusing on some environmental settings, development tools, programming and professional softwares.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/DerrickMarcus/DerrickMarcus.github.io.git
cd DerrickMarcus.github.io
```

Create a vitrual environment:

```bash
python -m venv blog_env
source blog_env/bin/activate  # Linux / macOS
blog_env\Scripts\activate  # Windows

pip install -r requirements.txt
```

Build the site:

```bash
mkdocs serve
```

Then open your browser and visit <http://127.0.0.1:8000/> to view the site.

It is recommended to add a file `.github/workflow/ci.yml` to use Github Actions to deploy the site.

## License

The documents of knowledge and technology is under [CC-BY-SA-4.0 license](./LICENSE).

The code is under [MIT license](./LICENSE_CODE).

## Contact

If you have any question or advice, please contact me by `blog@yanxuchen.com` .
