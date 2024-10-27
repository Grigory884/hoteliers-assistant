import logging
from telegram import Update, ForceReply, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from geopy.geocoders import Nominatim
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.rouge_score import rouge_n
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import openai
import sqlite3

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Функции для сбора данных ---

# Команда /start
def start(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    # Определение времени суток
    hour = int(update.message.date.hour)
    greeting = ""
    if 6 <= hour < 11:
        greeting = "Доброе утро, "
    elif 11 <= hour < 16:
        greeting = "Добрый день, "
    else:
        greeting = "Добрый вечер, "
    update.message.reply_html(
        rf"{greeting}{user.mention_html()}! Я бот-ассистент для создания описания Вашего отеля! Напишите мне название отеля, чтобы я смог приступить к обработке запроса.",
        reply_markup=ForceReply(selective=True),
    )

# Сбор данных о отеле
def scrape_hotel_info(hotel_name):
    hotel_info = {}
    # Поиск в Google
    search_url = f"https://www.google.com/search?q={hotel_name}+отель"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    return hotel_info

def extract_google_reviews(hotel_name):
    """
    Извлекает отзывы из Google Maps для указанного отеля.

    Args:
        hotel_name (str): Название отеля.
        location (str, optional): Местоположение отеля (город, район). Defaults to "Москва".

    Returns:
        list: Список словарей, где каждый словарь содержит:
            - review_text: Текст отзыва.
            - sentiment: Тональность отзыва (positive, negative, neutral).
    """
    reviews = []
    analyzer = SentimentIntensityAnalyzer()
    search_url = f"https://www.google.com/search?q={hotel_name}&tbm=lcl&hl=ru"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Найти элементы, содержащие отзывы
    review_elements = soup.find_all('div', class_='review-content')
    for review_element in review_elements:
        review_text = review_element.find('span', class_='review-text').text.strip()
        sentiment_scores = analyzer.polarity_scores(review_text)
        sentiment = 'positive' if sentiment_scores['compound'] >= 0.05 else 'negative' if sentiment_scores['compound'] <= -0.05 else 'neutral'
        reviews.append({'review_text': review_text, 'sentiment': sentiment})
    return reviews

def analyze_reviews(reviews):
    """
    Анализирует отзывы и выделяет ключевые особенности отеля.

    Args:
        reviews (list): Список словарей с отзывами и тональностью.

    Returns:
        dict: Словарь с ключевыми особенностями отеля, например:
            - positive_features: Положительные особенности (например, "удобное расположение", "чистота" , "шикарная кухня" , "дружелюбный персонал").
            - negative_features: Отрицательные особенности (например, "шумные соседи", "тонкие стены", "ужасный завтрак" , "грязные номера", "не вежливый персонал").
    """
    positive_features = []
    negative_features = []
    stop_words = set(stopwords.words('russian'))
    lemmatizer = WordNetLemmatizer()

    for review in reviews:
        if review['sentiment'] == 'positive':
            words = [lemmatizer.lemmatize(word.lower()) for word in review['review_text'].split() if word.lower() not in stop_words]
            positive_features.extend(words)
        elif review['sentiment'] == 'negative':
            words = [lemmatizer.lemmatize(word.lower()) for word in review['review_text'].split() if word.lower() not in stop_words]
            negative_features.extend(words)

    # Определение уникальных особенностей (настройка токенизации и лемматизации)
    unique_positive_features = set(positive_features)
    unique_negative_features = set(negative_features)

    return {
        'positive_features': unique_positive_features,
        'negative_features': unique_negative_features,
    }

def analyze_nearby_competitors(hotel_name, location="Москва", radius=1000):  # радиус в метрах
    """
    Анализирует конкурентов отеля в ближайшем радиусе, чтобы выделить его отличительные свойства.

    Args:
        hotel_name (str): Название отеля.
        location (str, optional): Местоположение отеля (город, район). Defaults to "Москва".
        radius (int, optional): Радиус поиска конкурентов в метрах. Defaults to 1000.

    Returns:
        dict: Словарь с отличительными свойствами отеля по сравнению с конкурентами, например:
            - unique_features: Уникальные свойства отеля.
            - competitor_features: Свойства конкурентов.
    """
    geolocator = Nominatim(user_agent="hotel_assistant")
    location_coords = geolocator.geocode(location)
    if location_coords:
        latitude = location_coords.latitude
        longitude = location_coords.longitude
        search_url = f"https://www.google.com/maps/search/{hotel_name}+{location}/@{latitude},{longitude},{radius}z/data=!3m1!4b1"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        competitor_elements = soup.find_all('div', class_='VkpGBb')
        competitor_names = [element.find('span', class_='LrzXr zdqRlf').text for element in competitor_elements]

        unique_features = set()
        competitor_features = set()

        for competitor_name in competitor_names:
            competitor_reviews = extract_google_reviews(competitor_name, location)
            competitor_features.update(analyze_reviews(competitor_reviews)['positive_features'])

        # Сравнить с особенностями анализируемого отеля
        hotel_reviews = extract_google_reviews(hotel_name, location)
        hotel_features = analyze_reviews(hotel_reviews)['positive_features']

        # Используем метрику Jaccard для определения уникальных свойств
        jaccard_similarity = len(hotel_features.intersection(competitor_features)) / len(hotel_features.union(competitor_features))
        if jaccard_similarity < 0.7:  # Если сходство меньше 70%, считаем, что отель имеет уникальные свойства
            unique_features = hotel_features - competitor_features

        return {
            'unique_features': unique_features,
            'competitor_features': competitor_features,
        }
    else:
        return {'unique_features': set(), 'competitor_features': set()}

def analyze_target_audience(hotel_name, location="Москва"):
    """
    Анализирует целевую аудиторию отеля на основе данных о его местоположении и типах услуг.

    Args:
        hotel_name (str): Название отеля.
        location (str, optional): Местоположение отеля (город, район). Defaults to "Москва".

    Returns:
        list: Список предполагаемых сегментов целевой аудитории (например, "Бизнесмены", "Семейные пары", "Молодеые люди").
    """

    # 1. Получить информацию о местоположении отеля (почтовый индекс, район, город)
    geolocator = Nominatim(user_agent="hotel_assistant")
    location_coords = geolocator.geocode(location)
    if location_coords:
        location_details = location_coords.raw
        postcode = location_details.get('address', {}).get('postcode')
        neighbourhood = location_details.get('address', {}).get('neighbourhood')
        city = location_details.get('address', {}).get('city')

    # 2. Собрать информацию о доступных услугах (например, с Google Maps или сайта отеля)
    search_url = f"https://www.google.com/search?q={hotel_name}+{location}&tbm=lcl&hl=ru"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    service_elements = soup.find_all('span', class_='LrzXr')
    services = [service_element.text.strip() for service_element in service_elements]

    # 3. Определить сегменты целевой аудитории на основе местоположения и услуг
    target_audience = []
    if postcode and "business" in postcode:
        target_audience.append("Бизнесмены")
    if neighbourhood and "family" in neighbourhood:
        target_audience.append("Семейные пары")
    if city and "youth" in city:
        target_audience.append("Молодые люди")
    if any(service in services for service in ["тренажерный зал", "бизнес-центр", "конференц-зал"]):
        target_audience.append("Бизнесмены")
    if any(service in services for service in ["детская площадка", "семейные номера", "услуги няни"]):
        target_audience.append("Семейные пары")
    if any(service in services for service in ["бар", "ночной клуб", "дискотека"]):
        target_audience.append("Молодые люди")
    return target_audience

# ---  Остальные функции ---

# Генерация описания с помощью GPT-3
def generate_hotel_description(hotel_name, reviews, features, unique_features, target_audience):
    """
    Генерирует текст описания отеля с помощью GPT-3, учитывая целевую аудиторию.

    Args:
        hotel_name (str): Название отеля.
        reviews (list): Список словарей с отзывами и тональностью.
        features (dict): Словарь с ключевыми особенностями отеля.
        unique_features (set): Уникальные свойства отеля.
        target_audience (list): Список сегментов целевой аудитории.

    Returns:
        str: Текст описания отеля.
    """
    openai.api_key = 'api_ключ'
    prompt = f"""
    Опишите отель {hotel_name} для {', '.join(target_audience)} с учетом следующих особенностей:

    Положительные: {', '.join(features['positive_features'])}
    Отрицательные: {', '.join(features['negative_features'])}
    Уникальные особенности: {', '.join(unique_features)}

    Включите в описание некоторые примеры из отзывов:

    {', '.join([review['review_text'] for review in reviews[:3]])}
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.7,
    )
    return response.choices[0].text

# Обработка текста от пользователя
def handle_message(update: Update, context: CallbackContext) -> None:
    hotel_name = update.message.text
    user_id = update.effective_user.id
    # Создание базы данных, если она еще не существует
    create_database()
    # Проверка, есть ли у пользователя запись в базе данных
    if not check_user_exists(user_id):
        # Добавление нового пользователя в базу данных
        add_new_user(user_id)
    google_reviews = extract_google_reviews(hotel_name)
    features = analyze_reviews(google_reviews)
    competitor_analysis = analyze_nearby_competitors(hotel_name)
    target_audience = analyze_target_audience(hotel_name)
    # Сохранение данных в user_data
    context.user_data['hotel_name'] = hotel_name
    context.user_data['google_reviews'] = google_reviews
    context.user_data['features'] = features
    context.user_data['competitor_analysis'] = competitor_analysis
    context.user_data['target_audience'] = target_audience
    # Создание эталонного текста для оценки качества
    # Используем GPT-3 для генерации эталонных текстов
    openai.api_key = 'ваш_api_ключ'
    prompt = f"""
    Сгенерируйте два варианта описания отеля {hotel_name}, используя следующие особенности:

    Положительные: {', '.join(features['positive_features'])}
    Отрицательные: {', '.join(features['negative_features'])}
    Уникальные особенности: {', '.join(competitor_analysis['unique_features'])}
    Целевая аудитория: {', '.join(target_audience)}

    Примеры отзывов:
    {', '.join([review['review_text'] for review in google_reviews[:3]])}
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.7,
    )
    reference_texts = response.choices[0].text.split('\n\n')[:2]  # Извлечение двух вариантов описания
    description = generate_hotel_description(hotel_name, google_reviews, features, competitor_analysis['unique_features'], target_audience)
    # Оценка качества текста
    smoothie = SmoothingFunction().method4  # Использование SmoothingFunction для предотвращения деления на ноль
    bleu_scores = [sentence_bleu(reference_text.split(), description.split(), smoothing_function=smoothie) for reference_text in reference_texts]
    average_bleu_score = sum(bleu_scores) / len(bleu_scores)
    # ROUGE-оценка
    rouge_scores = [rouge_n(reference_text.split(), description.split(), n=2) for reference_text in reference_texts]
    average_rouge_score = sum(rouge_scores) / len(rouge_scores)
    # METEOR-оценка
    meteor_score_value = meteor_score(reference_texts, description)
    # Добавьте  SARI (Sentence-level Automatic Ranking Indicator)
    #from nltk.translate.sari import sentence_sari
    #sari_score = sentence_sari(reference_texts, description)
    # Добавьте  TER (Translation Edit Rate) 
    #from nltk.translate.ter import ter
    #ter_score = ter(reference_texts, description)
    if description:
        # Показ описания пользователю
        update.message.reply_text(f"Описание отеля: {description}\n"
                                 f"BLEU score: {average_bleu_score}\n"
                                 f"ROUGE-2 score: {average_rouge_score}\n"
                                 f"METEOR score: {meteor_score_value}"
                                 #f"SARI score: {sari_score}\n"
                                 #f"TER score: {ter_score}\n"
                                 )
        # Создание меню с кнопками "Описание одобрено" и "Необходимо переработать"
        reply_keyboard = [['Описание одобрено', 'Необходимо переработать']]
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        update.message.reply_text("Как вам описание?", reply_markup=markup)
        return  # Ожидание ответа пользователя
# Обработка нажатия кнопки "Описание одобрено"
def handle_approved_description(update: Update, context: CallbackContext) -> None:
    # Создание меню с кнопками форматов
    reply_keyboard = [['HTML', 'TXT', 'PDF']]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    update.message.reply_text("В каком формате вы хотите получить описание?", reply_markup=markup)
# Обработка нажатия кнопки "Необходимо переработать"
def handle_rejected_description(update: Update, context: CallbackContext) -> None:
    hotel_name = context.user_data.get('hotel_name')  # Получение имени отеля из user_data
    google_reviews = context.user_data.get('google_reviews')
    features = context.user_data.get('features')
    competitor_analysis = context.user_data.get('competitor_analysis')
    target_audience = context.user_data.get('target_audience')
    # Создание меню с кнопками "Ручная переработка" и "Автоматическая переработка"
    reply_keyboard = [['Ручная переработка', 'Автоматическая переработка']]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    update.message.reply_text("Как вы хотите переработать описание?", reply_markup=markup)
# Обработка нажатия кнопки "Ручная переработка"
def handle_manual_edit(update: Update, context: CallbackContext) -> None:
    hotel_name = context.user_data.get('hotel_name')
    update.message.reply_text("Введите измененное описание:")
    # Ожидание ответа пользователя с измененным текстом
    # Обработчик для получения измененного текста
    def handle_edit_text(update: Update, context: CallbackContext) -> None:
        description = update.message.text
        context.user_data['description'] = description
        user_id = update.effective_user.id
        save_description(user_id, hotel_name, description, 'manual')  # Сохранение измененного описания в базу
        # Создание меню с кнопками "Описание одобрено" и "Необходимо переработать"
        reply_keyboard = [['Описание одобрено', 'Необходимо переработать']]
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        update.message.reply_text("Как вам переработанное описание?", reply_markup=markup)
    context.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_edit_text))
# Обработка нажатия кнопки "Автоматическая переработка"
def handle_automatic_edit(update: Update, context: CallbackContext) -> None:
    hotel_name = context.user_data.get('hotel_name')
    google_reviews = context.user_data.get('google_reviews')
    features = context.user_data.get('features')
    competitor_analysis = context.user_data.get('competitor_analysis')
    target_audience = context.user_data.get('target_audience')
    # Повторная генерация описания
    description = generate_hotel_description(hotel_name, google_reviews, features, competitor_analysis['unique_features'], target_audience)
    # Оценка качества текста
    smoothie = SmoothingFunction().method4  # Использование SmoothingFunction для предотвращения деления на ноль
    reference_texts = [
        f"Отель {hotel_name} - это отличное место для отдыха. Он предлагает {', '.join(features['positive_features'])}.",
        f"В отеле {hotel_name} вы найдете {', '.join(features['positive_features'])}."
    ]
    bleu_scores = [sentence_bleu(reference_text.split(), description.split(), smoothing_function=smoothie) for reference_text in reference_texts]
    average_bleu_score = sum(bleu_scores) / len(bleu_scores)
    # ROUGE-оценка
    rouge_scores = [rouge_n(reference_text.split(), description.split(), n=2) for reference_text in reference_texts]
    average_rouge_score = sum(rouge_scores) / len(rouge_scores)
    # METEOR-оценка
    meteor_score_value = meteor_score(reference_texts, description)
    # Показ нового описания
    update.message.reply_text(f"Переработанное описание отеля: {description}\n"
                             f"BLEU score: {average_bleu_score}\n"
                             f"ROUGE-2 score: {average_rouge_score}\n"
                             f"METEOR score: {meteor_score_value}")
    # Создание меню с кнопками "Описание одобрено" и "Необходимо переработать"
    reply_keyboard = [['Описание одобрено', 'Необходимо переработать']]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    update.message.reply_text("Как вам переработанное описание?", reply_markup=markup)
# Обработка выбора формата файла
def handle_format_choice(update: Update, context: CallbackContext) -> None:
    chosen_format = update.message.text
    hotel_name = context.user_data.get('hotel_name')
    description = context.user_data.get('description', generate_hotel_description(hotel_name, context.user_data['google_reviews'], context.user_data['features'], context.user_data['competitor_analysis']['unique_features'], context.user_data['target_audience'])) # Используем описание из user_data или перегенерируем
    user_id = update.effective_user.id
    if chosen_format == 'HTML':
        # Создание HTML-файла
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{hotel_name}</title>
        </head>
        <body>
            <h1>Описание отеля {hotel_name}</h1>
            <p>{description}</p>
        </body>
        </html>
        """
        bio = io.BytesIO(html_content.encode('utf-8'))
        bio.name = 'description.html'
        update.message.reply_document(document=bio)
        update.message.reply_text("Описание Вашего отеля успешно завершено, ждем Вас вновь!")
        # Сохранение описания в базу данных
        save_description(user_id, hotel_name, description, 'html')
    elif chosen_format == 'TXT':
        # Создание TXT-файла
        bio = io.BytesIO(description.encode('utf-8'))
        bio.name = 'description.txt'
        update.message.reply_document(document=bio)
        update.message.reply_text("Описание Вашего отеля успешно завершено, ждем Вас вновь!")
        # Сохранение описания в базу данных
        save_description(user_id, hotel_name, description, 'txt')
    elif chosen_format == 'PDF':
        # Создание PDF-файла
        bio = io.BytesIO()
        pdf = canvas.Canvas(bio, pagesize=letter)
        pdf.setFont("Helvetica", 12)
        pdf.drawString(100, 700, f"Описание отеля {hotel_name}")
        pdf.drawString(100, 650, description)
        pdf.save()
        bio.name = 'description.pdf'
        update.message.reply_document(document=bio)
        update.message.reply_text("Описание Вашего отеля успешно завершено, ждем Вас вновь!")
        # Сохранение описания в базу данных
        save_description(user_id, hotel_name, description, 'pdf')
def show_history(update: Update, context: CallbackContext) -> None:
    user_id = update.effective_user.id
    hotel_name = context.user_data.get('hotel_name')
    history = get_description_history(user_id, hotel_name)
    if history:
        history_text = "\n".join(f"**{timestamp}:**\n{description}" for description, timestamp in history)
        update.message.reply_text(f"История изменений описания отеля {hotel_name}:\n\n{history_text}")
    else:
        update.message.reply_text(f"История изменений для отеля {hotel_name} пуста.")
# --- Основная функция ---
def main() -> None:
    # Создание Updater и передача токена
    updater = Updater("TELEGRAM_BOT_TOKEN")
    # Получение диспетчера для регистрации обработчиков
    dispatcher = updater.dispatcher
    # Команды
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    # Обработчики кнопок
    dispatcher.add_handler(MessageHandler(Filters.regex('^Описание одобрено$'), handle_approved_description))
    dispatcher.add_handler(MessageHandler(Filters.regex('^Необходимо переработать$'), handle_rejected_description))
    dispatcher.add_handler(MessageHandler(Filters.regex('^(HTML|TXT|PDF)$'), handle_format_choice))
    dispatcher.add_handler(MessageHandler(Filters.regex('^Ручная переработка$'), handle_manual_edit))
    dispatcher.add_handler(MessageHandler(Filters.regex('^Автоматическая переработка$'), handle_automatic_edit))
    dispatcher.add_handler(CommandHandler("history", show_history))
    # Запуск бота
    updater.start_polling()
    updater.idle()
if __name__ == '__hoteliers-assistant__':
    hoteliers-assistant()

   