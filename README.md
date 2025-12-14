
# DDA5001-25fall Final Project

The final project code for the DDA5001-25fall. Please refer to the project manual for more detailed information.

## How to start

### Part 1

All files related to part 1 of the final project are located in `p1`, and thus, before you run any code, please make sure you are in the `p1` directory.

For part 1, if you do not have enough computing resources, you'd better follow the instruction for using free GPU resources from Kaggle, please refer to [this](./Kaggle_training.md).

The main entry point of part 1 is `p1/main.ipynb`, which is a Jupyter notebook. You can run it in your local machine or in Kaggle.

### Part 2

All files for Part 2 are located in the `p2` directory. Before running the code, please ensure you are in the correct working directory. For instance, as shown in `p2/main.ipynb`, you may need to change your working directory to `p2/src`.

As with Part 1, if you have limited computing resources, we recommend following the instructions for using free GPU resources on Kaggle, which can be found [here](./Kaggle_training.md).

For users with a compatible local machine, you can refer to `vllm_rollout.py` for accelerated inference using vLLM. (Please note that running vLLM on Kaggle can be challenging due to library and hardware compatibility issues.)

**Note:** `p2/example.ipynb` provides a minimal example of running LoRA on Kaggle. For simplicity, many features, such as detailed plots, have been omitted. Your final results should be more comprehensive.

While the core training logic should be preserved, you are encouraged to experiment with other components if you wish to explore further!

### Part 3

All files for Part 3 are located in the `p3` directory.

**Note:** `p3/example.ipynb` provides a partial example for reference.

#### Environment Setup

**Prerequisites**
Before configuring the Jupyter Notebook, ensure you have uploaded the **Part 3** code to Kaggle as a dataset. Please refer to [Kaggle_training.md](Kaggle_training.md) for detailed instructions.

**Environment Configuration**
To accelerate the inference process, this project utilizes **vLLM**, and 2 GPU cards with data parallelism.

In Kaggle main page, click "Create" button and select "Notebook", then, in the new notebook page, click "File" -> "Import Notebook" to upload `main.ipynb`.

After uploading your Jupyter Notebook to Kaggle, please configure the environment as follows:

1.  **Session Options:**
    *   **ACCELERATOR:** Set to `GPU T4 x2`.
    *   **INTERNET:** Set to `On`.
    *   **PERSISTENCE:** Set to `Files only` (allows files to persist across sessions).

2.  **Input Sources:**
    *   **Add your dataset:** Click "Add Input" and select the dataset containing your uploaded code.
    *   **Add vLLM environment:** Search for the following URL and add it as an input (this is required to install vLLM correctly):
        *   URL: [https://www.kaggle.com/datasets/yangjiahua/it-is-vllm-0-8-5](https://www.kaggle.com/datasets/yangjiahua/it-is-vllm-0-8-5)

**Local Development**
If you are running this on a local machine, please refer to [p3/requirements.txt](p3/requirements.txt) for the necessary environment dependencies.

## Acknowledgements

The project is based on [nanoGPT](https://github.com/karpathy/nanoGPT). Thanks for the great work!