import nbformat

# List of notebooks to merge
notebooks = ["baseline_resnet_mlp2.ipynb", "crnn_ctc_strategy3_draft1.ipynb", "cnn_transformer_ctc_clean.ipynb"]

merged_nb = nbformat.v4.new_notebook()
for nb_file in notebooks:
    with open(nb_file, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        merged_nb.cells.extend(nb.cells)

# Save merged notebook
with open("merged_notebook.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(merged_nb, f)