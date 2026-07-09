from recognition import recognize_face

nome, score = recognize_face(
    app,
    database,
    crop
)

print(nome)
print(score)