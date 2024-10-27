# Telegram Hotel Description Assistant Bot: Your All-in-One Hotel Description Helper
This Telegram bot is your ultimate companion for crafting captivating hotel descriptions that attract more guests. It leverages Google data, sentiment analysis, and OpenAI's powerful GPT-3 language model to streamline the process and deliver high-quality results. 
**Tired of spending hours writing compelling descriptions?**  Let this bot take the reins and generate captivating content for you!
## Key Features
* **Automated Description Generation:**  Say goodbye to writer's block! This bot generates multiple description options quickly and effortlessly, saving you time and effort.
* **Data-Driven Insights:**  Leveraging Google data, the bot analyzes hotel reviews, identifies nearby competitors, and even determines your target audience based on location and services. This ensures your descriptions are accurate, relevant, and tailored to the right people.
* **Targeted Descriptions:** The bot utilizes GPT-3 to create descriptions that resonate with specific audiences. Whether it's business travelers, families, or adventure seekers, the bot ensures your descriptions appeal to the right people.
* **Quality Evaluation:**  Get a clear picture of your descriptions' effectiveness with BLEU, ROUGE-2, and METEOR scores. These metrics help you gauge the quality of the generated text against reference descriptions.
* **Flexibility and Control:** You have complete control over the process.  You can:
    - **Manually Edit:**  Fine-tune the generated descriptions to your exact liking.
    - **Automatically Regenerate:** Have the bot create new versions based on your feedback.
    - **Choose Output Format:** Download the final description in HTML, TXT, or PDF formats for easy sharing.
* **History Tracking:**  The bot maintains a record of all generated and edited descriptions for each hotel, allowing you to easily revisit past versions.
## How to Use
1. **Obtain your Telegram Bot Token:**
   - Create a new bot in the Telegram BotFather ([https://t.me/BotFather](https://t.me/BotFather)).
   - Get your bot's unique token.
2. **Obtain your OpenAI API Key:**
   - Register for an OpenAI account: [https://platform.openai.com/](https://platform.openai.com/)
   - Create an API key. 
3. **Install Required Libraries:**
   ```bash
   pip install telegram==2.6.0 requests==2.31.0 beautifulsoup4==4.11.1 vaderSentiment==3.3.2 geopy==2.2.0 nltk==3.8.1 reportlab==3.6.10 openai==0.27.2 sqlite3
