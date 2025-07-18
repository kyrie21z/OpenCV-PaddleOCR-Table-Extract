from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="OCR")

output = pipeline.predict(
    input=".\page_13.png",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True,
)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")