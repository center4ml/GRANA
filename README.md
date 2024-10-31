# GRANA
<img src="https://img.shields.io/badge/Python-3.9-blue"/>
<a href="www.chloroplast.pl/GRANA"><img src="https://img.shields.io/badge/GRANA-Website-green" /></a>
<a href="https://huggingface.co/spaces/chloroplast/GRANA"><img src="https://img.shields.io/badge/GRANA-Demo-green" /></a>
<img src="https://img.shields.io/badge/Gradio-4.44.0-darkgreen"/>

GRANA (**G**raphical **R**ecognition and **A**nalysis of **N**anostructural **A**ssemblies) 
is an an AI-enhanced, user-friendly
software tool that recognizes grana on thylakoid network electron micrographs 
and generates a complex set of their structural parameters measurements.

## Website
More information about GRANA, including **example dataset**, can be found at [GRANA website](https://www.chloroplast.pl/grana).

## Demo
Demo version of GRANA is available at [Hugging Face Spaces](https://huggingface.co/spaces/chloroplast/GRANA).
Using demo version, you can analyze up to 5 images at once.

## Installation and usage

### Running as a pre-build Docker container
The recommended way to run GRANA is to use the Docker container form repository.

To run the container, use the following command:
```bash
docker run -p 7860:7860 mbuk/grana
```
After running the command, you can access the GRANA interface at `http://localhost:7860`.

### Building container locally
If you prefer to build the container locally, use the following commands:

1. Clone the repository:
    ```
    git clone git@github.com:center4ml/GRANA.git
    cd GRANA
    ```

2. Build the Docker image:
    ```
    docker build -t <image-name> .
    ```
    Where `<image-name>` is the name of your choice, e.g. "grana"

3. Run the Docker container:
    ```
    docker run -p 7860:7860 <image-name>
    ```

   After running the command, you can access the GRANA interface at `http://localhost:7860`.

### Running from Source

To run the project locally, ensure you have Python 3.9+ installed and follow these steps:

1. Clone the repository:
    ```
    git clone git@github.com:center4ml/GRANA.git
    cd GRANA
    ```

2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Start the Gradio interface:
    ```
    python app.py
    ```

The Gradio interface should now be running locally and accessible at http://localhost:7860.
