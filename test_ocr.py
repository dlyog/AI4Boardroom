from PIL import Image
import pytesseract

img = Image.open("static/uploads/087ba821-395b-401c-bf60-f6003d1aa670.png")
text = pytesseract.image_to_string(img)
print(repr(text))
