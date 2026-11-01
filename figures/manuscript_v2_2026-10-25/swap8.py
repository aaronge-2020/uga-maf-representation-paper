import json, subprocess, sys
from PIL import Image
CLI="/opt/hatch/bin/hatch_gws_cli"; DOC="1RDciU05KTkTsTKh-bOuJY0F_78_nWNhSL5fSC9WnFi0"
MC="/home/hatch/workspace/scratch/uga-maf-audit/manuscript-check"
def run(args, inp=None):
    r=subprocess.run([CLI]+args, input=inp, capture_output=True, text=True)
    if r.returncode!=0:
        print("FAIL", args[1:3], r.stderr[-600:], r.stdout[-300:]); sys.exit(1)
    return json.loads(r.stdout)
files={"figure1":"1eNGwYhSqzk5lFL122N4Nc5lM0Cc6uhFg","figure2":"1LH8QTKNbPVho-MolMTYLjdPj_mierNkU",
"figure3":"1Y0liHQyPdft0TgNs7tBTeYY4InOocvJg","figure4":"1bTB9McHGKYNvHr_KdWE1V7gWMLWlDaiu",
"figure5":"1M_L_jjq5OcNBL11MztEJo9_TikySf7DJ","figure6":"13s31S34MUueqvgTEDv_JNfAtchn0UA4U",
"figure7":"1258RAhetdLX6xmOfpXc9RNLrjvpkUlrI","figure8":"1xII5Wu1Scy0SKy88I3EbMqP-qM7bbJMp"}
widths={"figure1":432,"figure2":432,"figure3":432,"figure4":468,"figure5":468,"figure6":468,"figure7":432,"figure8":432}
order=["figure8","figure7","figure6","figure5","figure4","figure3","figure2","figure1"]
for name in order:
    d=run(["docs","documents","get","--params",json.dumps({"documentId":DOC,"includeTabsContent":True})])
    open(f"{MC}/gdoc_pre_{name}.json","w").write(json.dumps(d))
    images=[]
    def walk(el):
        for x in el.get("paragraph",{}).get("elements",[]):
            if x.get("inlineObjectElement"): images.append(x["startIndex"])
    for el in d["tabs"][0]["documentTab"]["body"]["content"]:
        if "paragraph" in el: walk(el)
        if "table" in el:
            for row in el["table"]["tableRows"]:
                for cell in row["tableCells"]:
                    for el2 in cell["content"]:
                        if "paragraph" in el2: walk(el2)
    img_ranges=sorted(set(images))
    idx={"figure1":0,"figure2":1,"figure3":2,"figure4":3,"figure5":4,"figure6":5,"figure7":6,"figure8":7}[name]
    start=img_ranges[idx]
    im=Image.open(f"/home/hatch/workspace/scratch/uga-maf-audit/figregen2/out/{name}.png")
    w=widths[name]; h=round(w*im.height/im.width,1)
    reqs=[{"deleteContentRange":{"range":{"startIndex":start,"endIndex":start+1}}},
          {"insertInlineImage":{"uri":f"https://drive.google.com/uc?id={files[name]}","location":{"index":start},
           "objectSize":{"width":{"magnitude":w,"unit":"PT"},"height":{"magnitude":h,"unit":"PT"}}}}]
    run(["docs","documents","batchUpdate","--params",json.dumps({"documentId":DOC}),
         "--json",json.dumps({"requests":reqs,"writeControl":{"requiredRevisionId":d["revisionId"]}})])
    print(name,"OK size",w,"x",h)
print("SWAP8 DONE")
