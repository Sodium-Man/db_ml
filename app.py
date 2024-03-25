from flask import Flask, render_template
import requests

app = Flask(__name__)

@app.route('/')
def index():
    # Define the API URL
    url = 'https://api.thingspeak.com/channels/2273418/feeds.json?api_key=ZJNP7M9ATK4IQAGJ&results=1000'

    # Send a GET request to the API URL
    response = requests.get(url)

    # Convert the response to JSON format
    data = response.json()

    # Extract the 'feeds' list from the JSON data
    feeds = data['feeds']

    # Extract data for the specified fields (Rain Status, Temperature, Humidity, Water Level, Soil Moisture)
    table_data = [[feed['field1'], feed['field2'], feed['field3'], feed['field4']] for feed in feeds]

    # Render the index.html template with the table data (in reverse order)
    return render_template('index.html', table_data=reversed(table_data))

if __name__ == '__main__':
    app.run(debug=True)
