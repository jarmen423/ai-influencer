import replicate

# Your character reference
face_ref = "character_refs/influencer_face.png"

# Scene prompt
prompt = "woman at desk with dual monitors, casual hoodie, candid photo"

# Call hosted model
output = replicate.run(
    "black-forest-labs/flux-1-dev-ipadapter",
    input={
        "prompt": prompt,
        "face_image": open(face_ref, "rb"),
        "num_outputs": 4
    }
)
