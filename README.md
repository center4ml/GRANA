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

## Running as Docker container
The recommended way to run GRANA is to use Docker container.

To run the container, use the following command:
```bash
docker run -p 7860:7860 mbuk/grana_measure:v0.5.4
```
After running the command, you can access the GRANA interface at `http://localhost:7860`.
