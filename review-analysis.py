import pandas as pd
import requests
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Используем ручной список стоп-слов
stop_words = [
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "то", "так", "ее", 
    "за", "бы", "по", "нет", "от", "к", "все", "она", "да", "себя", "этот", "его", "тоже", 
    "когда", "ли", "два", "эти", "один", "вот", "этот", "там", "такой", "для", "вы"
]

# Функция для сбора отзывов с сайта TripAdvisor
def get_reviews(url, headers, session):
    try:
        # Отправляем запрос с использованием сессии
        response = session.get(url)
        response.raise_for_status()  # Проверка на ошибки HTTP-запроса
        print(f"Запрос успешен: {url}")

        # Парсим HTML с помощью BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        reviews = []

        # Ищем все элементы с отзывами
        review_containers = soup.find_all('div', {'class': 'review-container'})

        for review in review_containers:
            # Извлекаем рейтинг
            rating = review.find('span', {'class': 'ui_bubble_rating'})
            if rating:
                rating = int(rating['class'][1].split('_')[1]) // 10  # Преобразуем рейтинг

            # Извлекаем текст отзыва
            text = review.find('p', {'class': 'partial_entry'})
            if text:
                text = text.get_text(strip=True)

            # Добавляем данные о рейтинге и отзыве в список
            reviews.append({"rating": rating, "text": text})

        # Возвращаем собранные данные в виде DataFrame
        return pd.DataFrame(reviews)

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return pd.DataFrame()  # Возвращаем пустой DataFrame в случае ошибки

# Заголовки для предотвращения блокировки (включая изменённый User-Agent)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

# Прокси (если есть, можете добавить сюда свои прокси)
proxies = {
    'http': 'http://your_proxy_here',
    'https': 'https://your_proxy_here',
}

# Создание сессии с прокси (если прокси используются)
session = requests.Session()
session.headers.update(headers)

# URL страницы с отзывами на TripAdvisor
url = "https://www.tripadvisor.ru/Hotel_Review-g187497-d189897-Reviews-Hotel_10_10-Barcelona_Catalonia.html"

# Попытка сбора данных
df = get_reviews(url, headers, session)

# Проверяем, что данные успешно собраны
if df.empty:
    print("Не удалось собрать данные с TripAdvisor.")
else:
    print(f"Собрано {len(df)} отзывов.")

    # Сохраняем собранные данные в файл CSV
    df.to_csv("hotel_reviews_tripadvisor.csv", index=False)

    # Проверка на наличие столбца 'text'
    if 'text' in df.columns:
        # Предобработка текста: нормализация, удаление стоп-слов и неалфавитных символов
        def preprocess_text(text):
            text = text.lower()  # Приведение к нижнему регистру
            text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text)  # Удаление всего, что не является буквой
            text = ' '.join([word for word in text.split() if word not in stop_words])  # Удаление стоп-слов
            return text

        # Применяем предобработку текста к отзывам
        df['text_clean'] = df['text'].apply(preprocess_text)

        # 1. Преобразование текста в векторное представление с использованием TF-IDF
        tfidf = TfidfVectorizer(max_features=3000, min_df=5)  # Ограничение на частоту слов
        X = tfidf.fit_transform(df['text_clean']).toarray()  # Применяем предобработку текста
        y = df['rating']  # Рейтинг отеля (целевой столбец)

        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Исследовательский анализ данных
        sns.countplot(x=df['rating'])
        plt.title("Распределение рейтингов отзывов")
        plt.xlabel("Рейтинг")
        plt.ylabel("Количество отзывов")
        plt.show()

        # Топ частотных слов в положительных/отрицательных отзывах
        positive_reviews = ' '.join(df[df['rating'] >= 4]['text_clean'])
        negative_reviews = ' '.join(df[df['rating'] <= 2]['text_clean'])

        # Визуализация частотных слов в положительных отзывах
        wordcloud = WordCloud(width=800, height=400, max_words=100).generate(positive_reviews)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Частотные слова в положительных отзывах")
        plt.show()

        # 3. Обучение моделей и оценка их точности
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)
        y_pred_log_reg = log_reg.predict(X_test)

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gbc.fit(X_train, y_train)
        y_pred_gbc = gbc.predict(X_test)

        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test)

        results = {
            "Логистическая регрессия": accuracy_score(y_test, y_pred_log_reg),
            "Случайный лес": accuracy_score(y_test, y_pred_rf),
            "Градиентный бустинг": accuracy_score(y_test, y_pred_gbc),
            "MLP (нейронная сеть)": accuracy_score(y_test, y_pred_mlp),
        }

        for model, accuracy in results.items():
            print(f"{model}: {accuracy:.4f}")

        # 4. Гиперпараметрическая оптимизация
        param_grid_rf = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
        grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
        grid_search_rf.fit(X_train, y_train)

        print("Лучшие параметры для случайного леса:", grid_search_rf.best_params_)
        print("Лучшая точность:", grid_search_rf.best_score_)

        # 5. Кросс-валидация
        cross_val_scores_rf = cross_val_score(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42), X, y, cv=5, scoring='accuracy')
        print("Средняя точность случайного леса на кросс-валидации:", cross_val_scores_rf.mean())

        # 6. Матрицы ошибок
        def plot_confusion_matrix(y_true, y_pred, model_name):
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
            plt.title(f"Матрица ошибок для {model_name}")
            plt.xlabel('Предсказанные значения')
            plt.ylabel('Истинные значения')
            plt.show()

        plot_confusion_matrix(y_test, y_pred_log_reg, "Логистическая регрессия")
        plot_confusion_matrix(y_test, y_pred_rf, "Случайный лес")
        plot_confusion_matrix(y_test, y_pred_gbc, "Градиентный бустинг")
        plot_confusion_matrix(y_test, y_pred_mlp, "Нейронная сеть (MLP)")

    else:
        print("Нет данных о текстах отзывов. Проверьте HTML-структуру страницы.")