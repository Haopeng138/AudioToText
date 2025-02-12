import re

f = open("algo.txt", "r")
text = f.read()

pattern = r"\[\d+\.\d+:\d+\.\d+\]\s*"

# Remove timestamps
clean_text = re.sub(pattern, "", text)

# write the cleaned text to a new file
with open("cleaned_algo.txt", "w") as f:
    f.write(clean_text)