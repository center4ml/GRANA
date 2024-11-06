import os
import shutil
import time
import uuid
from datetime import datetime
from decimal import Decimal

import gradio as gr
import matplotlib.pyplot as plt

from settings import DEMO

plt.switch_backend("agg")  # fix for "RuntimeError: main thread is not in main loop"
import numpy as np
import pandas as pd
from PIL import Image

from model import GranaAnalyser

ga = GranaAnalyser(
    "weights/detection_model.pt",
    "weights/orientation_model.ckpt",
    "weights/period_model.ckpt",
)


def calc_ratio(pixels, nano):
    """
    Calculates ratio of pixels to nanometers and returns as str to populate ratio_input
    :param pixels:
    :param nano:
    :return:
    """
    if not (pixels and nano):
        pass
    else:
        res = pixels / nano
        return res


# https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
def KDE(dataset, h):
    # the Kernel function
    def K(x):
        return np.exp(-(x ** 2) / 2) / np.sqrt(2 * np.pi)

    n_samples = dataset.size

    x_range = dataset  # x-value range for plotting KDEs

    total_sum = 0
    # iterate over datapoints
    for i, xi in enumerate(dataset):
        total_sum += K((x_range - xi) / h)

        y_range = total_sum / (h * n_samples)

    return y_range


def prepare_files_for_download(
    dir_name,
    grana_data,
    aggregated_data,
    detection_visualizations_dict,
    images_grana_dict,
):
    """
    Save and zip files for download
    :param dir_name:
    :param grana_data: DataFrame containing all grana measurements
    :param aggregated_data: dict containing aggregated measurements
    :return:
    """
    dir_to_zip = f"{dir_name}/to_zip"

    # raw data
    grana_data_csv_path = f"{dir_to_zip}/grana_raw_data.csv"
    grana_data.to_csv(grana_data_csv_path, index=False)

    # aggregated measurements
    aggregated_csv_path = f"{dir_to_zip}/grana_aggregated_data.csv"
    aggregated_data.to_csv(aggregated_csv_path)

    # annotated pictures
    masked_images_dir = f"{dir_to_zip}/annotated_images"
    os.makedirs(masked_images_dir)
    for img_name, img in detection_visualizations_dict.items():
        filename_split = img_name.split(".")
        extension = filename_split[-1]
        filename = ".".join(filename_split[:-1])
        filename = f"{filename}_annotated.{extension}"
        img.save(f"{masked_images_dir}/{filename}")

    # single_grana images
    grana_images_dir = f"{dir_to_zip}/single_grana_images"
    os.makedirs(grana_images_dir)
    org_images_dict = pd.Series(
        grana_data["source image"].values, index=grana_data["granum ID"]
    ).to_dict()
    for img_name, img in images_grana_dict.items():
        org_filename = org_images_dict[img_name]
        org_filename_split = org_filename.split(".")
        org_filename_no_ext = ".".join(org_filename_split[:-1])
        img_name_ext = f"{org_filename_no_ext}_granum_{str(img_name)}.png"
        img.save(f"{grana_images_dir}/{img_name_ext}")

    # zip all files
    date_str = datetime.today().strftime("%Y-%m-%d")
    zip_name = f"GRANA_results_{date_str}"
    zip_path = f"{dir_name}/{zip_name}"
    shutil.make_archive(zip_path, "zip", dir_to_zip)

    # delete to_zip dir
    zip_dir_path = os.path.join(os.getcwd(), dir_to_zip)
    shutil.rmtree(zip_dir_path)

    download_file_path = f"{zip_path}.zip"
    return download_file_path


def show_info_on_submit(s):
    return (
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.Row(visible=True),
        gr.Row(visible=False),
    )


def load_css():
    with open("styles.css", "r") as f:
        css_content = f.read()
    return css_content


primary_hue = gr.themes.Color(
    c50="#e1f8ee",
    c100="#b7efd5",
    c200="#8de6bd",
    c300="#63dda5",
    c400="#39d48d",
    c500="#27b373",
    c600="#1e8958",
    c700="#155f3d",
    c800="#0c3522",
    c900="#030b07",
    c950="#000",
)


theme = gr.themes.Default(
    primary_hue=primary_hue,
    font=[gr.themes.GoogleFont("Ubuntu"), "ui-sans-serif", "system-ui", "sans-serif"],
)


def draw_violin_plot(y, ylabel, title):
    # only generate plot for 3 or more values
    if y.count() < 3:
        return None

    # Colors
    RED_DARK = "#850e00"
    DARK_GREEN = "#0c3522"
    BRIGHT_GREEN = "#8de6bd"

    # Create jittered version of "x" (which is only 1)
    x_jittered = []
    kde = KDE(y, (y.max() - y.min()) / y.size / 2)
    kde = kde / kde.max() * 0.2
    for y_val in kde:
        x_jittered.append(1 + np.random.uniform(-y_val, y_val, 1))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=x_jittered, y=y, s=20, alpha=0.4, c=DARK_GREEN)

    violins = ax.violinplot(
        y,
        widths=0.45,
        bw_method="silverman",
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # change violin color
    for pc in violins["bodies"]:
        pc.set_facecolor(BRIGHT_GREEN)

    # add a boxplot to ax
    # but make the whiskers length equal to 1 SD, i.e. in the proportion of the IQ range, but this length should start from the mean but be visible from the box boundary
    lower = np.mean(y) - 1 * np.std(y)
    upper = np.mean(y) + 1 * np.std(y)

    medianprops = dict(linewidth=1, color="black", solid_capstyle="butt")
    boxplot_stats = [
        {
            "med": np.median(y),
            "q1": np.percentile(y, 25),
            "q3": np.percentile(y, 75),
            "whislo": lower,
            "whishi": upper,
        }
    ]

    ax.bxp(
        boxplot_stats,  # data for the boxplot
        showfliers=False,  # do not show the outliers beyond the caps.
        showcaps=True,  # show the caps
        medianprops=medianprops,
    )

    # Add mean value point
    ax.scatter(1, y.mean(), s=30, color=RED_DARK, zorder=3)

    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    return fig


def transform_aggregated_results_table(results_dict):
    MEASUREMENT_HEADER = "measurement [unit]"
    VALUE_HEADER = "value +-SD"

    def get_value_str(value, std):
        if np.isnan(value) or np.isnan(std):
            return "-"
        value_str = str(Decimal(str(value)).quantize(Decimal("0.01")))
        std_str = str(Decimal(str(std)).quantize(Decimal("0.01")))
        return f"{value_str} +-{std_str}"

    def append_to_dict(new_key, old_val_key, old_sd_key):
        aggregated_dict[MEASUREMENT_HEADER].append(new_key)
        value_str = get_value_str(results_dict[old_val_key], results_dict[old_sd_key])
        aggregated_dict[VALUE_HEADER].append(value_str)

    aggregated_dict = {MEASUREMENT_HEADER: [], VALUE_HEADER: []}

    # area
    append_to_dict("area [nm^2]", "area nm^2", "area nm^2 std")

    # perimeter
    append_to_dict("perimeter [nm]", "perimeter nm", "perimeter nm std")

    # diameter
    append_to_dict("diameter [nm]", "diameter nm", "diameter nm std")

    # height
    append_to_dict("height [nm]", "height nm", "height nm std")

    # number of layers
    append_to_dict("number of thylakoids", "Number of layers", "Number of layers std")

    # SRD
    append_to_dict("SRD [nm]", "period nm", "period nm std")

    # GSI
    append_to_dict("GSI", "GSI", "GSI std")

    # N grana
    aggregated_dict[MEASUREMENT_HEADER].append("number of grana")
    aggregated_dict[VALUE_HEADER].append(str(int(results_dict["N grana"])))

    return aggregated_dict


def rename_columns_in_results_table(results_table):
    column_names = {
        "Granum ID": "granum ID",
        "File name": "source image",
        "area nm^2": "area [nm^2]",
        "perimeter nm": "perimeter [nm]",
        "diameter nm": "diameter [nm]",
        "height nm": "height [nm]",
        "Number of layers": "number of thylakoids",
        "period nm": "SRD [nm]",
        "period SD nm": "SRD SD [nm]",
    }
    results_table = results_table.rename(columns=column_names)
    return results_table


with gr.Blocks(css=load_css(), theme=theme) as demo:

    svg = """
<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30.73 33.38">
  <defs>
    <style>
      .cls-1 {
        fill: #27b373;
        stroke-width: 0px;
      }
    </style>
  </defs>
  <path class="cls-1" d="M19.69,11.73h-3.22c-2.74,0-4.96,2.22-4.96,4.96h0c0,2.74,2.22,4.96,4.96,4.96h3.43c.56,0,1,.51.89,1.09-.08.43-.49.72-.92.72h-8.62c-.74,0-1.34-.6-1.34-1.34v-10.87c0-.74.6-1.34,1.34-1.34h13.44c2.73,0,4.95-2.22,4.95-4.95h0c0-2.75-2.22-4.97-4.96-4.97h-13.85C4.85,0,0,4.85,0,10.83v11.71c0,5.98,4.85,10.83,10.83,10.83h9.07c5.76,0,10.49-4.52,10.81-10.21.35-6.29-4.72-11.44-11.02-11.44ZM19.9,31.4h-9.07c-4.89,0-8.85-3.96-8.85-8.85v-11.71C1.98,5.95,5.95,1.98,10.83,1.98h13.81c1.64,0,2.97,1.33,2.97,2.97h0c0,1.65-1.33,2.97-2.96,2.97h-13.4c-1.83,0-3.32,1.49-3.32,3.32v10.87c0,1.83,1.49,3.32,3.32,3.32h8.56c1.51,0,2.83-1.12,2.97-2.62.16-1.72-1.2-3.16-2.88-3.16h-3.52c-1.64,0-2.97-1.33-2.97-2.97h0c0-1.64,1.33-2.97,2.97-2.97h3.34c4.83,0,8.9,3.81,9.01,8.64s-3.9,9.04-8.84,9.04Z"/>
  <path class="cls-1" d="M19.9,29.41h-9.07c-3.79,0-6.87-3.07-6.87-6.87v-11.71c0-3.79,3.07-6.87,6.87-6.87h13.81c.55,0,.99.44.99.99h0c0,.55-.44.99-.99.99h-13.81c-2.7,0-4.88,2.19-4.88,4.88v11.71c0,2.7,2.19,4.88,4.88,4.88h8.94c2.64,0,4.91-2.05,5-4.7s-2.12-5.05-4.87-5.05h-3.52c-.55,0-.99-.44-.99-.99h0c0-.55.44-.99.99-.99h3.36c3.74,0,6.9,2.92,7.01,6.66.11,3.87-3.01,7.06-6.85,7.06Z"/>
</svg>
"""

    gr.HTML(
        f'<div class="header"><div id="header-logo">{svg}</div><div id="header-text">GRANA<div></div>'
    )

    with gr.Row(elem_classes="input-row"):  # input
        with gr.Column():
            gr.HTML(
                "<h1>1. Choose images to upload. All the images need to be of the same scale and experimental variant.</h1>"
            )
            img_input = gr.File(file_count="multiple")

            gr.HTML("<h1>2. Set the scale of the images for the measurements.</h1>")
            with gr.Row():
                with gr.Column():
                    gr.HTML("Either provide pixel per nanometer ratio...")
                    ratio_input = gr.Number(
                        label="pixel per nm", precision=3, step=0.001
                    )

                with gr.Column():
                    gr.HTML("...or length of the scale bar in pixels and nanometers.")
                    pixels_input = gr.Number(label="Length in pixels")
                    nano_input = gr.Number(label="Length in nanometers")

                    pixels_input.change(
                        calc_ratio,
                        inputs=[pixels_input, nano_input],
                        outputs=ratio_input,
                    )
                    nano_input.change(
                        calc_ratio,
                        inputs=[pixels_input, nano_input],
                        outputs=ratio_input,
                    )

            with gr.Row():
                clear_btn = gr.ClearButton(img_input, "Clear")
                submit_btn = gr.Button("Submit", variant="primary")

    with gr.Row(visible=False) as loading_row:
        with gr.Column():
            gr.HTML(
                "<div class='processed-info'>Images are being processed. This may take a while...</div>"
            )

    with gr.Row(visible=False) as output_row:
        with gr.Column():
            gr.HTML(
                '<div class="results-header">Results</div>'
                "<p>Full results are a zip file containing:<p>"
                "<ul>- grana_raw_data.csv: a table with full grana measurements,</ul>"
                "<ul>- grana_aggregated_data.csv: a table with aggregated measurements,</ul>"
                '<ul>- directory "annotated_images" with all submitted images with masks on detected grana,</ul>'
                '<ul>- directory "single_grana_images" with images of all detected grana.</ul>'
                "<p>Note that GRANA only stores the result files for 1 hour.</p>",
                elem_classes="input-row",
            )
            with gr.Row(elem_classes="input-row"):
                download_file_out = gr.DownloadButton(
                    label="Download results",
                    variant="primary",
                    elem_classes="margin-bottom",
                )
            with gr.Row():
                gr.HTML(
                    '<h2 class="title">Annotated images</h2>'
                    "Gallery of uploaded images with masks of recognized grana structures. "
                    "Each granum mask is "
                    "labeled with its number. Note that only fully visible grana in the image are masked."
                )
            with gr.Row(elem_classes="margin-bottom"):
                gallery_out = gr.Gallery(
                    columns=4,
                    rows=2,
                    object_fit="contain",
                    label="Detection visualizations",
                    show_download_button=False,
                )

            with gr.Row(elem_classes="input-row"):
                gr.HTML(
                    '<h2 class="title">Aggregated results for all uploaded images</h2>'
                )
            with gr.Row(elem_classes=["input-row", "margin-bottom"]):
                table_out = gr.Dataframe(label="Aggregated data")

            with gr.Row():
                gr.HTML(
                    '<h2 class="title">Violin graphs</h2>'
                    "These graphs present aggregated results for selected structural parameters. "
                    "The graph for each parameter is only generated if three or more values are available. "
                    "Each graph "
                    "displays individual data points, a box plot indicating the first and third quartiles, whiskers "
                    "marking the standard deviation (SD), the median value (horizontal line on the box plot), "
                    "the mean value (red dot), and a density plot where the width represents the frequency."
                )

            with gr.Row():
                area_plot_out = gr.Plot(label="Area")
                perimeter_plot_out = gr.Plot(label="Perimeter")
                gsi_plot_out = gr.Plot(label="GSI")

            with gr.Row(elem_classes="margin-bottom"):
                diameter_plot_out = gr.Plot(label="Diameter")
                height_plot_out = gr.Plot(label="Height")
                srd_plot_out = gr.Plot(label="SRD")

            with gr.Row():
                gr.HTML(
                    '<h2 class="title">Recognized and rotated grana structures</h2>'
                )

            with gr.Row(elem_classes="margin-bottom"):
                gallery_single_grana_out = gr.Gallery(
                    columns=4,
                    rows=2,
                    object_fit="contain",
                    label="Single grana images",
                    show_download_button=False,
                )

            with gr.Row():
                gr.HTML(
                    '<h2 class="title">Full results</h2>'
                    "Note that structural parameters other than area and perimeter are only calculated for the grana "
                    "whose direction and/or SRD could be estimated."
                )
            with gr.Row():
                table_full_out = gr.Dataframe(label="Full measurements data")

    submit_btn.click(
        show_info_on_submit,
        inputs=[submit_btn],
        outputs=[submit_btn, clear_btn, loading_row, output_row],
    )

    def enable_submit():
        return (
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Row(visible=False),
        )

    def gradio_analize_image(images, scale):
        """
        Model accepts following parameters:
        :param images: list of images to be processed, in either tiff or png format
        :param scale: float, nm to pixel ratio

        Model returns the following objects:
        - detection_visualizations: list of images with masks to be displayed as gallery and served to download
        as zip of images
        - grana_data: dataframe with measurements for each image to be served to download as a csv file
        - images_grana: list of images with single grana to be served to download as zip of images
        - aggregated_data: dataframe with aggregated measurements for all images to be displayed as table and served
        to download as csv
        """

        # validate that at least one image has been uploaded
        if images is None or len(images) == 0:
            raise gr.Error("Please upload at least one image")

        # on demo instance, we limit the number of images to 5
        if DEMO:
            if len(images) > 5:
                raise gr.Error("In demo version it is possible to analyze up to 5 images.")

        # validate that scale has been provided correctly
        if scale is None or scale == 0:
            raise gr.Error("Please provide scale. Use dot as decimal separator")

        # validate that all images are png or tiff
        for image in images:
            if not image.name.lower().endswith((".png", ".tif", ".jpg", ".jpeg")):
                raise gr.Error("Only png, tiff, jpg ang jpeg images are supported")

        # clean up previous results
        # find all directories in current working directory that start with "results_"
        # that were created more than 1 hour ago and delete them with all contents
        for directory_name in os.listdir():
            if directory_name.startswith("results_"):
                dir_path = os.path.join(os.getcwd(), directory_name)
                if os.path.isdir(dir_path):
                    if time.time() - os.path.getctime(dir_path) > 60 * 60:
                        shutil.rmtree(dir_path)

        # create a directory for results
        results_dir_name = "results_{uuid}".format(uuid=uuid.uuid4().hex)
        os.makedirs(results_dir_name)
        zip_dir_name = f"{results_dir_name}/to_zip"
        os.makedirs(zip_dir_name)

        # model takes a dict of images, so we need to convert input to list of PIL.PngImagePlugin.PngImageFile or
        # PIL.TiffImagePlugin.TiffImageFile objects
        images_dict = {
            image.name.split("/")[-1]: Image.open(image.name)
            for i, image in enumerate(images)
        }

        # model works here
        (
            detection_visualizations_dict,
            grana_data,
            images_grana_dict,
            aggregated_data,
        ) = ga.predict(images_dict, scale)
        detection_visualizations = list(detection_visualizations_dict.values())
        images_grana = list(images_grana_dict.values())

        # rearrange aggregated data to be displayed as table
        aggregated_dict = transform_aggregated_results_table(aggregated_data)
        aggregated_df_transposed = pd.DataFrame.from_dict(aggregated_dict)

        # rename columns in full results
        grana_data = rename_columns_in_results_table(grana_data)

        # save files returned by model to disk so they can be retrieved for downloading
        download_file_path = prepare_files_for_download(
            results_dir_name,
            grana_data,
            aggregated_df_transposed,
            detection_visualizations_dict,
            images_grana_dict,
        )

        # generate plot
        area_fig = draw_violin_plot(
            grana_data["area [nm^2]"].dropna(),
            "Granum area [nm^2]",
            "Grana areas from all uploaded images",
        )
        perimeter_fig = draw_violin_plot(
            grana_data["perimeter [nm]"].dropna(),
            "Granum perimeter [nm]",
            "Grana perimeters from all uploaded images",
        )
        gsi_fig = draw_violin_plot(
            grana_data["GSI"].dropna(),
            "GSI",
            "GSI from all uploaded images",
        )
        diameter_fig = draw_violin_plot(
            grana_data["diameter [nm]"].dropna(),
            "Granum diameter [nm]",
            "Grana diameters from all uploaded images",
        )
        height_fig = draw_violin_plot(
            grana_data["height [nm]"].dropna(),
            "Granum height [nm]",
            "Grana heights from all uploaded images",
        )
        srd_fig = draw_violin_plot(
            grana_data["SRD [nm]"].dropna(), "SRD [nm]", "SRD from all uploaded images"
        )

        return [
            gr.Row(visible=True),
            gr.Row(visible=True),
            download_file_path,
            detection_visualizations,
            aggregated_df_transposed,
            area_fig,
            perimeter_fig,
            gsi_fig,
            diameter_fig,
            height_fig,
            srd_fig,
            images_grana,
            grana_data,
        ]

    submit_btn.click(
        fn=gradio_analize_image,
        inputs=[
            img_input,
            ratio_input,
        ],
        outputs=[
            loading_row,
            output_row,
            # file_download_checkboxes,
            download_file_out,
            gallery_out,
            table_out,
            area_plot_out,
            perimeter_plot_out,
            gsi_plot_out,
            diameter_plot_out,
            height_plot_out,
            srd_plot_out,
            gallery_single_grana_out,
            table_full_out,
        ],
    ).then(fn=enable_submit, inputs=[], outputs=[submit_btn, clear_btn, loading_row])

demo.launch(
    share=False, debug=True, server_name="0.0.0.0", allowed_paths=["images/logo.svg"]
)
