Objective: Developed a multi-label classification system to accurately classify anime into its corresponding genre(s) using diverse data sources from MyAnimeList.

Approach and Techniques:
 - Metadata Analysis: Employed a Fully Connected Network (FCN) to process metadata such as episode count, airing seasons, years, directors, and producers, achieving an average accuracy of 73%.
 - Poster Analysis: Implemented a pre-trained ResNet model to classify anime based on visual posters, yielding an average accuracy of 78%.
 - Name and Synopsis Analysis: Utilized BERT to generate embeddings from the anime's name and synopsis, which were then processed through an FCN, resulting in an average accuracy of 81%.
 - Combined Approach: Integrated data from the synopsis, name, and posters, achieving the highest accuracy of 84%.

Key Takeaway: Demonstrated that classifying anime into genres using the name alone is feasible, laying a solid foundation for future advancements in genre classification.
