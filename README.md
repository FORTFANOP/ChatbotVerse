# ChatbotVerse
A completely easy to use beginner-friendly Python module that lets you whip up your own custom chatbot with just a few lines of code. This AI-based package feeds your custom data into a neural network which is trained and outputs a fully functional chatbot.

## Installation
You can install it from Pypi:
`pip install ChatbotVerse`
https://pypi.org/project/ChatbotVerse

Here's how easy it is to instantiate a neural network based bot using ChatbotVerse:

### Importing the module
`from ChatbotVerse import chatbotVerse as cbv`

### Initialize trainer
```
trainer = cbv.modelTrain()
intents = trainer.loadIntents('intents.json')  # The path where the intents.json file is saved
words, classes = trainer.preprocess_save_Data(intents)  # Prepares and saves preprocessed word data
train_x, train_y = trainer.prepareTrainingData(words, classes)  # Prepares training data
```

### Create the model
`model = trainer.createModel(train_x, train_y, save_path='cbv_model.model')`

### Initialize predictor
`predictor = cbv.modelPredict('intents.json', 'cbv_model.model')`

### Get output from the bot
```
running = True
while running:
    msg = input('You: ')
    if msg == 'quit':
        running = False
    else:
        response = predictor.chatbot_response(msg)
        print('Bot: ', response)
```
