from paddlex import create_model
model = create_model("PP-LCNet_x1_0_doc_ori")
output = model.predict("textline_rot180_demo.jpg",  batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/demo.png")
    res.save_to_json("./output/res.json")