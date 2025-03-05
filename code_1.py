# Data cleaning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import nltk
amazon = pd.read_csv('amazon.csv')
# Restructure the data
amazon['discounted_price'] = amazon['discounted_price'].str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)
amazon['rating_count'] = amazon['rating_count'].str.replace(',', '', regex=True).astype(float)
amazon['discount_percentage'] = amazon['discount_percentage'].str.replace('%','',regex=True).astype(float)
amazon['rating'] = pd.to_numeric(amazon['rating'], errors='coerce')
amazon['actual_price'] = amazon ['actual_price'].str.replace('₹','',regex=True).str.replace(',','',regex=True).astype(float)
amazon['main_category'] = amazon['category'].str.split('|').str[0]
amazon['main_category'].describe()
amazon['sub_category'] = amazon['category'].str.split('|').str[-1]
# Check for missing values
amazon.info()
\
# Drop missing values
amazon.dropna(subset=['rating_count'],inplace=True)
amazon.dropna(subset=['rating'],inplace=True)

#Check for outliners
amazon.describe()
amazon.hist()

#Check for duplicates
amazon.drop_duplicates(inplace=True)

#Analysis

#key statistics
main_product = amazon.groupby('main_category').agg({
    'discount_percentage': 'mean',  # Mean discount percentage
    'actual_price': 'mean',         # Mean actual price
    'rating_count': 'sum',
    'rating': 'mean',
    'discounted_price': 'mean',
    })
main_product = main_product.sort_values(by='discount_percentage', ascending=False)

sub_product = amazon.groupby('sub_category').agg({
    'discount_percentage': 'mean',  # Mean discount percentage
    'actual_price': 'mean',         # Mean actual price
    'rating_count': 'sum',
    'rating': 'mean'
})
sub_product = sub_product.sort_values(by='discount_percentage', ascending=False)

print(main_product)
print(sub_product)
main_product = main_product.reset_index()
sub_product = sub_product.reset_index()
main_category_counts = amazon['main_category'].value_counts()
main_category_counts


#Discount effectiveness

#Bar chart show the number of product correspond to main category
bars=plt.bar(main_category_counts.index, main_category_counts.values, color='#ff9900')
plt.ylabel('Number of Products')
plt.title('Distribution of Products by Main Categories',fontweight="bold")
plt.xticks(rotation=90)
plt.gca().axes.get_yaxis().set_visible(False)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')
plt.show()

#Create a bar chart to show the average discount amount considering actual price
major_categories = ['Electronics','Computers&Accessories', 'Home&Kitchen', 'OfficeProducts']
yaprice = [10128, 1687, 1900, 397]
ydp=[5965, 845,899,301]
plt.figure()
x = np.arange(len(major_categories))  # X-axis locations
bar_width=0.2
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - bar_width, yaprice, width=bar_width, label='Actual price', color='#37475A')
ax.bar(x, ydp, width=bar_width, label='Discounted price', color='#FEBD69')
for i, v in enumerate(yaprice):
    ax.text(i - bar_width, v + 50, str(v), ha='center',size='8')  # Adjust vertical offset (50) as needed
for i, v in enumerate(ydp):
    ax.text(i, v + 50, str(v), ha='center',size='8')  # Adjust vertical offset (50) as needed
ax.set_xticks(x)
ax.set_xticklabels(major_categories)  # Set x-axis labels
ax.set_title("Actual vs Discounted Price among main categories",fontweight='bold')
ax.legend()
ax.set_yticks([])
ax.set_yticklabels([])
plt.show()

#Boxplot to show the distribution of discount perentage among main categories
df_major = amazon[amazon['main_category'].isin(major_categories)]
plt.figure(figsize=(12, 10))
fig, ax = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'hspace': 0.5})
df_major.boxplot(
      ax=ax[0],
    column="discount_percentage",
    by="main_category",
    patch_artist=True,
    boxprops=dict(facecolor="#FF9900", color="#FFB500"),
    medianprops=dict(color="#232F3E", linewidth=2),
    capprops=dict(color="#232F3E"),
    whiskerprops=dict(color="#FF9900"),
    flierprops=dict(marker='o', color="#FF9900", alpha=0.6),
    rot=90)
fig.suptitle("")
ax[0].set_title("Discount Distribution Across Major Categories",fontweight="bold")
ax[0].set_xlabel("Category", fontsize=10, color="#232F3E")
ax[0].set_ylabel("Discount Percentage", fontsize=10, color='#FEBD69')
ax[0].tick_params(axis='x', labelcolor="#232F3E", labelsize=9, rotation=0)
ax[0].tick_params(axis='y', labelcolor="#232F3E", labelsize=9)
ax[0].grid(False)
plt.show()

#Histogram to show the distribution of discount percentage with average rating
from scipy.interpolate import make_interp_spline
# Define discount bins
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Create histogram
fig, ax1 = plt.subplots(figsize=(8, 5))  # Create figure and axes for histogram
counts, edges, bars = ax1.hist(amazon['discount_percentage'], bins=bins, color='#FEBD69')
# Add count numbers on top of each column
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, count + 0.1, int(count), ha='center', va='bottom')
# Remove the y-axis on the left
ax1.set_ylabel("")  # Remove label
ax1.set_yticks([])  # Remove ticks
# Calculate average rating for each bin
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
avg_ratings_per_bin = []
for i in range(len(bins) - 1):
    bin_data = amazon[(amazon['discount_percentage'] >= bins[i]) & (amazon['discount_percentage'] < bins[i+1])]
    if len(bin_data) > 0:
        avg_ratings_per_bin.append(bin_data['rating'].mean())
    else:
        avg_ratings_per_bin.append(np.nan)
# Create spline interpolation for smoother curve
x_smooth = np.linspace(min(bin_centers), max(bin_centers), 300)
spl = make_interp_spline(bin_centers, avg_ratings_per_bin, k=3)
y_smooth = spl(x_smooth)
ax2 = ax1.twinx()
ax2.plot(x_smooth, y_smooth, color='red', label='Average Rating (Smoothed)')
ax2.set_ylim(3, 5)
ax2.set_ylabel("Average Rating", color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax1.set_xlabel("Discount Percentage") # Added ax1.set_xlabel
ax1.set_title("Discount Distribution with Average Rating",fontweight='bold') #Added ax1.set_title
ax1.set_xticks(bins) # Set the x-axis ticks using bins, which are numeric values
ax1.set_xticklabels([f'{b}%' for b in bins]) # Set the labels accordingly


#Bar chart to show the average actual price and discount amount by discount percentage bins
# Define discount bins and labels
bins = [0, 20, 40, 60, 80, 100]
labels = ["<20%", "20%-40%", "40%-60%", "60%-80%", ">80%"]
# Calculate actual discount value
amazon['actual_discount_value'] = amazon['actual_price'] - amazon['discounted_price']
# Bin the discount percentages
amazon['discount_bin'] = pd.cut(amazon['discount_percentage'], bins=bins, labels=labels, right=False)
# Calculate average actual price and discount amount per bin
discount_stats = amazon.groupby('discount_bin').agg({
    'actual_price': 'mean',
    'actual_discount_value': 'mean'
})
# Create bar chart
bar_width = 0.35
x = np.arange(len(discount_stats))
fig, ax = plt.subplots(figsize=(10, 6))
actual_price_bars = ax.bar(x - bar_width/2, discount_stats['actual_price'], bar_width,
                           label='Avg. Actual Price', color='#37475A')
discount_amount_bars = ax.bar(x + bar_width/2, discount_stats['actual_discount_value'], bar_width,
                              label='Avg. Discount Amount', color='#FF9900')
# Add values on top of bars
for bar in actual_price_bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
for bar in discount_amount_bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
# Customize chart
ax.set_title('Average Actual Price and Discount Amount by Discount Percentage Bins', fontweight='bold')
ax.set_xlabel("Discount Percentage Bins")
ax.set_yticks([]) # Remove y ticks
ax.set_yticklabels([]) #Remove y tick labels
ax.set_ylabel("") # Remove y label
ax.set_xticks(x)
ax.set_xticklabels(discount_stats.index, rotation=45)
ax.legend()
plt.tight_layout()
plt.show()

#Scatter plot to show the relationship between discount on rating count and rating
# --- Linear Regression ---
import statsmodels.api as sm
# Prepare the data for regression
amazon['log_rating_count'] = np.log(amazon['rating_count'])
X = amazon['discount_percentage']  # Independent variable
y =  amazon['log_rating_count'] # Dependent variable
# Add a constant to the independent variable (for the intercept)
X = sm.add_constant(X)
# Fit the linear regression model
model = sm.OLS(y, X).fit()
# Get the regression line values
predictions = model.predict(X)
# --- End of Linear Regression ---
# Scatter plot with regression line
fig, ax = plt.subplots(figsize=(10, 6)) #create a new figure and axes for this plot.
ax.scatter(amazon['discount_percentage'], amazon['log_rating_count'], color='#FF9900', edgecolors='#232F3E', alpha=0.6)
ax.plot(amazon['discount_percentage'], predictions, color='#37475A', linewidth=2)  # Plot the regression line
ax.set_title("Impact of Discounts on Log Rating Counts", fontsize=12, color="#232F3E", fontweight="bold")
ax.set_xlabel("Discount Percentage", fontsize=10, color="#232F3E", fontweight="bold")
ax.set_ylabel("Log Rating Counts", fontsize=10, color="#232F3E", fontweight="bold")
ax.tick_params(axis='x', labelcolor="#232F3E", labelsize=9, rotation=0)
ax.tick_params(axis='y', labelcolor="#232F3E", labelsize=9)
# Display model summary
print(model.summary())
plt.show()

# Prepare the data for regression
X = amazon['discount_percentage']  # Independent variable
y =  amazon['rating'] # Dependent variable
# Add a constant to the independent variable (for the intercept)
X = sm.add_constant(X)
# Fit the linear regression model
model = sm.OLS(y, X).fit()
# Get the regression line values
predictions = model.predict(X)
# --- End of Linear Regression ---
# Scatter plot with regression line
fig, ax = plt.subplots(figsize=(10, 6)) #create a new figure and axes for this plot.
ax.scatter(amazon['discount_percentage'], amazon['rating'], color='#FF9900', edgecolors='#232F3E', alpha=0.6)
ax.plot(amazon['discount_percentage'], predictions, color='#37475A', linewidth=2)  # Plot the regression line
# --- Linear Regression ---
ax.set_title("Impact of Discounts on Rating", fontsize=12, color="#232F3E", fontweight="bold")
ax.set_xlabel("Discount Percentage", fontsize=10, color="#232F3E", fontweight="bold")
ax.set_ylabel("Rating", fontsize=10, color="#232F3E", fontweight="bold")
ax.tick_params(axis='x', labelcolor="#232F3E", labelsize=9, rotation=0)
ax.tick_params(axis='y', labelcolor="#232F3E", labelsize=9)
# Display model summary
print(model.summary())
plt.show()

#Bar chart to show the rating count changes according to discount percentage among main categories
Discount_20_40 = [3935808.0, 1348790.0, 883218.0, 13728.0]
discount_40_60 = [1726549.0, 1196207.0, 659691.0, 10694.0]
discount_60_80 = [2942886.0, 3123743.0, 3663.0, 10959.0]
x = np.arange(len(major_categories))  # X-axis locations
bar_width = 0.2
# Create figure
fig, ax = plt.subplots(figsize=(7, 4))
# Plot bars side by side
ax.bar(x - bar_width, Discount_20_40, width=bar_width, label='Discount 20%-40%', color='#ff9900')
ax.bar(x, discount_40_60, width=bar_width, label='Discount 40%-60%', color='#FEBD69')
ax.bar(x + bar_width, discount_60_80, width=bar_width, label='Discount 60%-80%', color='#37475A')
# Formatting
ax.set_xticks(x)
ax.set_xticklabels(major_categories)  # Set x-axis labels
ax.set_title("Rating count changes according to discount percentage among main categories", fontweight='bold',fontsize=9)
# Calculate and add percentage change relative to 20-40% discount
for i in range(len(major_categories)):
    # Percentage change calculation (relative to 20-40%)
    perc_change_40_60 = ((discount_40_60[i] - Discount_20_40[i]) / Discount_20_40[i]) * 100
    perc_change_60_80 = ((discount_60_80[i] - Discount_20_40[i]) / Discount_20_40[i]) * 100
    ax.text(x[i] - bar_width, Discount_20_40[i], f'{0:.1f}%', ha='center', va='bottom',fontsize=6)  # 20-40% is the baseline (0%)
    ax.text(x[i], discount_40_60[i], f'{perc_change_40_60:.1f}%', ha='center', va='bottom',fontsize=6)
    ax.text(x[i] + bar_width, discount_60_80[i], f'{perc_change_60_80:.1f}%', ha='center', va='bottom',fontsize=6)
    ax.set_yticks([]) # Remove y ticks
ax.legend()
plt.show()

#Bar chart to show the rating changes according to discount percentage among main categories
Discount_20_40 =[4.13,4.2,4.09,4.37]
discount_40_60 = [4.09, 4.23, 3.99, 4.35]
discount_60_80 = [4.04,4.14, 3.92,4.1]
x = np.arange(len(major_categories))  # X-axis locations
bar_width = 0.25
# Create figure
fig, ax = plt.subplots(figsize=(7, 4))
# Plot bars side by side
ax.bar(x - bar_width, Discount_20_40, width=bar_width, label='Discount 20%-40%', color='#ff9900')
ax.bar(x, discount_40_60, width=bar_width, label='Discount 40%-60%', color='#FEBD69')
ax.bar(x + bar_width, discount_60_80, width=bar_width, label='Discount 60%-80%', color='#37475A')
# Formatting
ax.set_xticks(x)
ax.set_xticklabels(major_categories,fontsize=8)  # Set x-axis labels
ax.set_title("Impact of Discount percentage on Rating ", fontweight='bold',fontsize=10)
ax.legend()
#Adding values on top of bars
for i, v in enumerate(Discount_20_40):
    ax.text(i - bar_width, v + 0.05, str(v), ha='center')
for i, v in enumerate(discount_40_60):
    ax.text(i, v + 0.05, str(v), ha='center')
for i, v in enumerate(discount_60_80):
    ax.text(i + bar_width, v + 0.05, str(v), ha='center')
ax.set_yticks([])
ax.set_yticklabels([])
plt.show()

#Rating and sentiment analysis

#Rating distribution
# Adjust bins for better distribution (you can experiment with different bins)
bins = [0, 3, 4, 5]  # Changed bins to capture more variation
labels = ['<3', "3-4", '4-5']
amazon['Rating'] = pd.cut(amazon['rating'], bins=bins, labels=labels, right=False)
# Count occurrences of each bin
rating_count = amazon['Rating'].value_counts().sort_index()
# Create the pie chart
plt.pie(rating_count.values, labels=rating_count.index, autopct='%1.1f%%', startangle=90, explode=[0.03, 0.03, 0.03], colors=[ '#37475A','#FEBD69','#ff9900']) #added explode and colors
plt.title('Rating Distribution',fontweight='bold')
plt.show()

# Key words in reviews
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Assuming amazon DataFrame is defined elsewhere in your code
text_data = " ".join(amazon['review_content'].astype(str).tolist())

# Tokenization and cleaning
nltk.download('all')
tokens = word_tokenize(text_data)
tokens = [word.lower() for word in tokens if word.isalnum()]
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]

# Frequency distribution
fdist = FreqDist(tokens)

# Create bigrams and find the most frequent ones
from nltk import bigrams
bigram_dist = FreqDist(bigrams(tokens))
top_keyphrases = bigram_dist.most_common(20)

# Plotting
keyphrases, frequencies = zip(*top_keyphrases)
keyphrases_str = [' '.join(kp) for kp in keyphrases]

plt.figure(figsize=(12, 6))
plt.bar(keyphrases_str, frequencies, color='#37475A')
plt.title('Top 20 Keyphrase Frequencies in Review Content', fontweight='bold', fontsize=15)
plt.ylabel('Frequency')
plt.xticks(rotation=90, fontsize=12)
plt.tight_layout()
plt.show()

# Filter the dataframe to include only products with a rating lower than 3
low_rating_df = amazon[amazon['rating'] < 3]
# Create a string of all the reviews for these products
reviews_text = ' '.join(low_rating_df['review_content'].dropna().values)
# Generate the wordcloud
wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10,colormap='Oranges').generate(reviews_text)
# Plot the wordcloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
#Reviewers analysis
# Use the .str.split() command to split string values in the user_id and user_name columns using , as a delimiter.
# expand = False will insert every split string element into a list.
split_user_id = amazon['user_id'].str.split(',', expand = False)
split_user_name = amazon['user_name'].str.split(',', expand = False)
# Use the .explode() command to split each element of a list into a row.
# Note: Although each element in the list is in a different row, the elements share the same index number.
# Example: A list with 5 values will be split into 5 rows, but each of those 5 rows will have the same index number.
id_rows = split_user_id.explode()
name_rows = split_user_name.explode()
# Use the DataFrame() command to create a dataframe using the exploded lists.
df_id_rows = pd.DataFrame(id_rows)
df_name_rows = pd.DataFrame(name_rows)
# Use the .reset_index() command to reset the index so that each row has its own index number.
df_id_rows = df_id_rows.reset_index(drop = True)
df_name_rows = df_name_rows.reset_index(drop = True)
# Use the .merge() command to merge 2 dataframes together.
reviewers = pd.merge(df_id_rows, df_name_rows, left_index = True, right_index = True)
reviewers_count = reviewers['user_name'].value_counts().reset_index(name='counts')

fig, ax = plt.subplots(1, 1, figsize=(15, 7.5))

sns.barplot(data=reviewers_count.sort_values('counts', ascending=False).head(10), color='#ff9900',
            x='counts',
            y='user_name')
ax.set_title('Top 10 Reviewers', fontsize=15, fontweight='bold') # Set title with fontsize and fontweight

# Show number for each row, delete x tick, x label, y label
for i in ax.containers:
    ax.bar_label(i,)
ax.set_xticklabels([])
ax.set_xlabel('')
plt.show()