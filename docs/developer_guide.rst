Developer Guide
===============

We welcome contributions to CYCLOPS! Whether you are fixing bugs, improving documentation, or adding new scientific features, your help is appreciated.

Improving the Heme Model
------------------------

Currently, the most significant area for potential contribution is the development of a **more realistic Heme force field model**. 

While the current implementation uses custom constraints for Heme-Fe-N coordination, moving toward a more sophisticated model (such as a bonded model with refined parameters or a non-bonded model with improved electronic descriptions) is a high priority for the project's evolution.

How to Contribute
-----------------

1. **Fork the Repository:** Create a personal fork of the `cyclops` repository on GitHub.
2. **Clone and Install:** Follow the installation instructions in the :doc:`getting_started` guide, ensuring you install in "editable" mode (``pip install -e .``).
3. **Create a Branch:** Use a descriptive name for your branch (e.g., ``feature/improved-heme-ff``).
4. **Submit a Pull Request:** Once your changes are tested, submit a PR for review.
