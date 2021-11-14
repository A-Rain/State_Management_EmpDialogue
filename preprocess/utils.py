import re

def simple_process_text(text: str):
    # replace _comma_ into ,
    sent = text.replace("_comma_", ",")

    sent = sent.replace("I;m", "I'm")
    sent = re.sub("Im ", " I\'m ", sent)
    sent = re.sub(" im ", " I\'m ", sent)

    sent = re.sub(" i ", " I ", sent)

    sent = re.sub("Ive ", "I\'ve ", sent)
    sent = re.sub(" ive ", " I\'ve ", sent)

    sent = re.sub("Ill ", "I'll ", sent)

    sent = re.sub(" didnt ", " didn\'t ", sent)
    sent = re.sub("Didnt ", " Didn\'t ", sent)

    sent = re.sub(" couldnt ", " couldn\'t ", sent)
    sent = re.sub("Couldnt ", "Couldn\'t ", sent)

    sent = re.sub(" wouldnt ", " wouldn\'t ", sent)
    sent = re.sub("Wouldnt ", "Wouldn\'t ", sent)

    sent = re.sub(" cant ", " can\'t ", sent)
    sent = re.sub("Cant ", "Can\'t ", sent)

    sent = re.sub(" wasnt ", " wasn\'t ", sent)
    sent = re.sub(" Wasnt ", " Wasn\'t ", sent)

    sent = re.sub(" werent ", " weren\'t ", sent)
    sent = re.sub("Werent", "Weren\'t", sent)

    sent = re.sub(" hasnt ", " hasn\'t ", sent)
    sent = re.sub("Hasnt", "Hasn\'t", sent)

    sent = re.sub("Havent", "Haven\'t", sent)
    sent = re.sub(" havent ", " haven\'t ", sent)

    sent = re.sub(" hadnt ", " hadn\'t ", sent)
    sent = re.sub("Hadnt ", "Hadn\'t ", sent)

    sent = re.sub("Isnt ", "Isn\'t ", sent)
    sent = re.sub(" isnt ", " isn\'t ", sent)

    sent = re.sub(" arent ", " aren\'t ", sent)
    sent = re.sub(" Arent ", " Aren\'t ", sent)

    sent = re.sub("Wont ", "Won\'t ", sent)
    sent = re.sub(" wont ", " won\'t ", sent)

    sent = re.sub(" doesnt ", " doesn\'t ", sent)
    sent = re.sub("Doesnt ", "Doesn\'t ", sent)

    sent = re.sub("Dont ", "Don\'t ", sent)
    sent = re.sub(" dont ", " don\'t ", sent)

    sent = re.sub("Thats ", "That\'s ", sent)
    sent = re.sub(" thats ", " that\'s ", sent)

    return sent