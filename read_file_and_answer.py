from transformers import pipeline

#.\myenv\scripts\activate.ps1

nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

xx = nlp(
    "https://templates.invoicehome.com/invoice-template-us-neat-750px.png",
    "What is the invoice number?"
)

print(xx)

# xx = nlp(
#     "https://miro.medium.com/max/787/1*iECQRIiOGTmEFLdWkVIH2g.jpeg",
#     "What is the purchase amount?"
# )

# print(xx)