# Python File Reader Template

Template for a custom Python-based file reader that hooks into *OVITO* and can easily be shared with other users.

This repository contains a template for creating your own [Python file reader](https://ovito.org/docs/dev/python/introduction/custom_file_readers.html), which can be installed into *OVITO Pro* or the [`ovito`](https://pypi.org/project/ovito/) Python module using *pip*.

## Getting Started

1. Click the "Use this template" button to create your own repository based on this template.
2. Rename `src/FileReaderName` to reflect the name of your file reader.
3. Implement your [file reader](https://ovito.org/docs/dev/python/introduction/custom_file_readers.html) in [`src/FileReaderName/__init__.py`](src/FileReaderName/__init__.py). Fill in the predefined functions as needed. More details on this interface can be found in the [OVITO Python docs](https://ovito.org/docs/dev/python/modules/ovito_io.html#ovito.io.FileReaderInterface). 
4. Fill in the [`pyproject.toml`](pyproject.toml) file. Fields that need to be replaced with your information are enclosed in descriptive `[[field]]` tags. Please make sure to include ovito>=3.9.1 as a dependency. Depending on your needs, you can add additional fields to the `pyproject.toml` file. Information can be found [here](https://setuptools.pypa.io/en/latest/userguide/index.html).
5. Fill in the [`README_Template.md`](README_Template.md) file. Again, the `[[fields]]` placeholders should guide you. Feel free to add other sections like "Images", "Citation", or "References" as needed.
6. Add meaningful examples and data sample files to the `Examples` directory to help others understand the use of your file reader.
7. Pick a license for your project and replace the current (MIT) [`LICENSE`](LICENSE) file with your license. If you keep the MIT license, please update the name and year in the current file.
8. Once you're done, rename `README_Template.md` to `README.md`, replacing this file.

## Testing
This repository is configured to enable automated testing using the [pytest](https://docs.pytest.org/en/7.4.x/) framework. Tests are automatically executed after each push to the main branch. To set up and activate automated testing, follow these two steps:

1. Write your tests in the `test/test_file_reader.py` file. You can also use other filenames that adhere to the pytest requirements.
2. Open the `.github/workflows/python-tests.yml` file and remove the `if: ${{ false }}` condition on line 15.

If needed, you can also adjust the operating system and Python versions by modifying the following lines:
```yaml
os: [ubuntu-latest, macos-latest, windows-latest]
python-version: ["3.7", "3.8", "3.9", "3.10", "3.11"]
```

An example can be found [here](https://github.com/ovito-org/GenerateRandomSolution).

As of August 16, 2023, according to the [GitHub documentation](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions), *"GitHub Actions usage is free for standard GitHub-hosted runners in public repositories, and for self-hosted runners."* Please refer to the GitHub documentation if you are uncertain about incurring costs.
