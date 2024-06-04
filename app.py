import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import emoji

st.sidebar.title("Intellichat-AI")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    st.dataframe(df)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # Finding the busiest users in the group (Group level)
        if selected_user == 'Overall':
            st.title('Most Busy users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most common words
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most common words')
        st.pyplot(fig)

        # Emoji Analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2) 

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df["Frequency"].head(), labels=emoji_df["Emoji"].head(), autopct="%0.2f")
            st.pyplot(fig)

        # User Sentiment Analysis
        st.title("Sentiment Analysis")

        # Perform sentiment analysis for the selected user or overall
        user_sentiments = helper.analyze_sentiment(selected_user, df)

        # If the selected user is 'Overall', display sentiment for all users
        if selected_user == 'Overall':
            overall_sentiment = {key: sum([user[key] for user in user_sentiments.values()]) / len(user_sentiments) for key in ['pos', 'neu', 'neg', 'compound']}
            st.subheader("Overall")

            # Create a horizontal bar chart for the overall sentiment scores
            fig, ax = plt.subplots()
            ax.barh(list(overall_sentiment.keys()), list(overall_sentiment.values()), color=['green' if v >= 0 else 'red' for v in overall_sentiment.values()])
            ax.set_xlabel('Score')
            ax.set_title('Overall sentiment scores')
            st.pyplot(fig)
        else:
            # If the selected user exists in the sentiment analysis results
            if selected_user in user_sentiments:
                sentiment_scores = user_sentiments[selected_user]

                st.subheader(f"User: {selected_user}")

                # Create a horizontal bar chart for the sentiment scores
                fig, ax = plt.subplots()
                ax.barh(list(sentiment_scores.keys()), list(sentiment_scores.values()), color=['green' if v >= 0 else 'red' for v in sentiment_scores.values()])
                ax.set_xlabel('Score')
                ax.set_title(f'Sentiment scores for {selected_user}')
                st.pyplot(fig)
            else:
                st.write(f"No sentiment analysis results for {selected_user}")

        # Final results for the sentiment analysis
        sentiment = helper.sentiment_score(user_sentiments, selected_user)
        st.subheader(f"The sentiment of {selected_user} is {sentiment}")
