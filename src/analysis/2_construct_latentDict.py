import pandas as pd
import random
import json
from vllm import LLM, SamplingParams
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
sampling_params = SamplingParams(temperature=0.6,top_p=0.9, max_tokens = 64) # 

def getLlamaResult(conversation):
    outputs = llm.chat(messages=conversation,
                   sampling_params=sampling_params,
                   use_tqdm=False)

    generated_text = outputs[0].outputs[0].text
    # print_outputs(outputs)
    return generated_text



exp_example_seq_dict = {
	"Grocery_and_Gourmet_Food":'''<start>
Hodgson Mill Gluten Free Yellow Cake Mix, 15-Ounce (Pack of 6)  0
Jennie's Coconut Macaroons, Peanut and Gluten Free, 8-Ounce Canisters (Pack of 6)       0
Schar Table Crackers Gluten Free, 7.4-Ounce (Pack of 3) 0
Peeled Snacks Apple-2-the-core, 1.23-Ounce Boxes (Pack of 10)   0
Kellogg's Harvest Acres Fruit Snacks, Mixed Fruit, 8 Ounce (Pack of 10) 10
<end>
<start>
Nawgan Mandarin Orange, 11.5-Ounce (Pack of 24) 0
Just Chill Tropical,  12 Ounce (Pack of 12)     0
Gratify Gluten Free Pretzel Sticks, 14.1 Ounce (Pack of 6)      0
Orville Redenbacher's Ready to Eat Popcorn, Brown Sugar Cinnamon, 6 Ounce       0
Gratify Gluten Free Pretzel Twists, 14.1 Ounce (Pack of 6)      0
Kellogg's Harvest Acres Fruit Snacks, Mixed Fruit, 8 Ounce (Pack of 10) 0
Cheez-It Zingz Wafer Queso, Fundito, 12.4 Ounce 0
Kellogg's Fruity Snacks, Mixed Berry, 16 Ounce (Pack of 6)      0
Peeled Snacks Apple-2-the-core, 1.23-Ounce Boxes (Pack of 10)   0
Keebler El Duende Cookies, Coconut, 11 Ounce (Pack of 12)       10
<end>
<start>
Hodgson Mill Brownie Mix with Whole Wheat Flour &amp; Milled Flax Seed, 12-Ounce Units (Pack of 6)      0
Hodgson Mill Whole Wheat Wild Blueberry Muffin Mix, 10-Ounce Boxes (Pack of 6)  0
KIND Fruit &amp; Nut Bar, Almond &amp; Coconut, 1.4-Ounce Bars (Pack of 8)      0
KIND Fruit &amp; Nut Bar, Almond &amp; Apricot, 1.4-Ounce Bars (Pack of 12)     0
Bob's Red Mill Organic Scottish Oatmeal, 20-Ounce Bags (Pack of 4)      0
Funky Monkey Snacks Carnaval Mix, Freeze-Dried Fruit, 1-Ounce Bags (Pack of 12) 0
Mrs. May's  Trio Bar Variety Pack, 1.2-Ounce bars (Pack of 20)  0
Mrs. May's Strawberry Trio Bar, 1.2-Ounce Bars (Pack of 24)     10
<end>''',

# 	"ML_1MTOPK":'''<start>
# Star Wars: Episode V - The Empire Strikes Back (1980)\t0\nDr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)\t0\nDelicatessen (1991)\t0\nClockwork Orange, A (1971)\t0\nBlade Runner (1982)\t0\nTwelve Monkeys (1995)\t0\nTerminator, The (1984)\t0\nE.T. the Extra-Terrestrial (1982)\t0\nContact (1997)\t0\nSneakers (1992)\t0\nPi (1998)\t0\nUsual Suspects, The (1995)\t0\nFargo (1996)\t0\nSilence of the Lambs, The (1991)\t0\nManchurian Candidate, The (1962)\t0\nTaxi Driver (1976)\t0\nNikita (La Femme Nikita) (1990)\t0\nFugitive, The (1993)\t0\nReservoir Dogs (1992)\t0\nSeven (Se7en) (1995)\t0\nSimple Plan, A (1998)\t4
# <end>
# <start>
# Star Wars: Episode V - The Empire Strikes Back (1980)\t0\nDr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)\t0\nDelicatessen (1991)\t0\nClockwork Orange, A (1971)\t0\nBlade Runner (1982)\t0\nTwelve Monkeys (1995)\t0\nTerminator, The (1984)\t0\nE.T. the Extra-Terrestrial (1982)\t0\nContact (1997)\t0\nSneakers (1992)\t0\nPi (1998)\t0\nUsual Suspects, The (1995)\t0\nFargo (1996)\t0\nSilence of the Lambs, The (1991)\t0\nManchurian Candidate, The (1962)\t0\nTaxi Driver (1976)\t0\nNikita (La Femme Nikita) (1990)\t0\nFugitive, The (1993)\t0\nReservoir Dogs (1992)\t0\nSeven (Se7en) (1995)\t0\nSimple Plan, A (1998)\t4\n<end>\n<start>\nOrdinary People (1980)\t0\nDie Hard 2 (1990)\t0\nSlums of Beverly Hills, The (1998)\t0\nGodfather, The (1972)\t0\nUnforgiven (1992)\t0\nGandhi (1982)\t0\nSnow White and the Seven Dwarfs (1937)\t0\nOutlaw Josey Wales, The (1976)\t0\nTotal Recall (1990)\t0\nBasic Instinct (1992)\t0\nMoll Flanders (1996)\t0\nAmerican Pie (1999)\t0\nGodfather: Part II, The (1974)\t0\n8 1/2 (1963)\t0\nJoy Luck Club, The (1993)\t0\nSpeed (1994)\t0\nJackie Brown (1997)\t0\nGrand Canyon (1991)\t0\nThelma & Louise (1991)\t0\nMetropolis (1926)\t0\nDr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)\t7
# <end>''',\
'ML_1MTOPK':'''<start>\nClockwork Orange, A (1971) (Sci-Fi)\t0\n2001: A Space Odyssey (1968) (Drama, Mystery, Sci-Fi, Thriller)\t0\nPlanet of the Apes (1968) (Action, Sci-Fi)\t0\nThing, The (1982) (Action, Horror, Sci-Fi, Thriller)\t0\nTotal Recall (1990) (Action, Adventure, Sci-Fi, Thriller)\t0\nTime Bandits (1981) (Adventure, Fantasy, Sci-Fi)\t0\nX-Men (2000) (Action, Sci-Fi)\t0\nMen in Black (1997) (Action, Adventure, Comedy, Sci-Fi)\t0\nGladiator (2000) (Action, Drama)\t0\nPatriot, The (2000) (Action, Drama, War)\t6\n<end>
<start>\nGiant (1956) (Drama)\t0\nSunset Blvd. (a.k.a. Sunset Boulevard) (1950) (Film-Noir)\t0\nDouble Indemnity (1944) (Crime, Film-Noir)\t0\nL.A. Confidential (1997) (Crime, Film-Noir, Mystery, Thriller)\t0\nFrankenstein (1931) (Horror)\t0\nRosemary's Baby (1968) (Horror, Thriller)\t0\nNosferatu (Nosferatu, eine Symphonie des Grauens) (1922) (Horror)\t0\nKing Kong (1933) (Action, Adventure, Horror)\t0\nWest Side Story (1961) (Musical, Romance)\t0\nMy Fair Lady (1964) (Musical, Romance)\t6\n<end>
<start>\nX-Men (2000) (Action, Sci-Fi)\t0\nStar Wars: Episode V - The Empire Strikes Back (1980) (Action, Adventure, Drama, Sci-Fi, War)\t0\nStar Wars: Episode IV - A New Hope (1977) (Action, Adventure, Fantasy, Sci-Fi)\t0\nAnna and the King (1999) (Drama, Romance)\t0\nCivil Action, A (1998) (Drama)\t0\nBug's Life, A (1998) (Animation, Children's, Comedy)\t0\nChicken Run (2000) (Animation, Children's, Comedy)\t0\nDouble Jeopardy (1999) (Action, Thriller)\t0\nGladiator (2000) (Action, Drama)\t0\nFrequency (2000) (Drama, Thriller)\t6\n<end>
<start>\nFight Club (1999) (Drama)\t0\nHidden, The (1987) (Action, Horror, Sci-Fi)\t0\nGo (1999) (Crime)\t0\nThere's Something About Mary (1998) (Comedy)\t0\nU Turn (1997) (Action, Crime, Mystery)\t0\nAdventures of Buckaroo Bonzai Across the 8th Dimension, The (1984) (Adventure, Comedy, Sci-Fi)\t0\nFunny Face (1957) (Comedy, Musical)\t0\nSense and Sensibility (1995) (Drama, Romance)\t0\nTrue Romance (1993) (Action, Crime, Romance)\t0\nBig Lebowski, The (1998) (Comedy, Crime, Mystery, Thriller)\t6\n<end>
<start>\nKing of the Hill (1993) (Drama)\t0\nFly Away Home (1996) (Adventure, Children's)\t0\nAmerican Beauty (1999) (Comedy, Drama)\t0\nSound of Music, The (1965) (Musical)\t0\nBasketball Diaries, The (1995) (Drama)\t0\nFree Enterprise (1998) (Comedy, Romance, Sci-Fi)\t0\nMarathon Man (1976) (Thriller)\t0\n12 Angry Men (1957) (Drama)\t0\nWest Side Story (1961) (Musical, Romance)\t0\nMy Fair Lady (1964) (Musical, Romance)\t7\n<end>''',\
# "LastFM":
# '''<start>
# I Like It\t0
# Say So\t10
# <end>
# <start>
# Reasons I Drink\t0
# Stupid Love\t8
# <end>
# <start>
# Bags\t0
# One More Year\t7
# <end>
# <start>
# Someone You Loved\t0
# Don't Start Now\t8
# <end>
# <start>
# Esperar Pra Ver\t0
# Lucro (Descomprimindo)\t9
# <end>'''
# "LastFM":'''<start>
# I Like It, in album Invasion of Privacy, by Cardi B\t0
# Say So, in album Hot Pink, by Doja Cat\t10
# <end>
# <start>
# Reasons I Drink, in album Reasons I Drink, by Alanis Morissette\t0
# Stupid Love, in album Stupid Love, by Lady Gaga\t8
# <end>
# <start>
# Bags, in album Bags, by Clairo\t0
# One More Year, in album The Slow Rush, by Tame Impala\t7
# <end>
# <start>
# Someone You Loved, in album Divinely Uninspired to a Hellish Extent, by Lewis Capaldi\t0
# Don't Start Now, in album Don't Start Now, by Dua Lipa\t8
# <end>
# <start>
# Esperar Pra Ver, in album Brazilian Compilation Series, Vol. 2, by Poolside\t0
# Lucro (Descomprimindo), in album Duas Cidades, by BaianaSystem\t9
# <end>'''
"LastFM":'''<start>
2 Wicky, in album A New Stereophonic Sound Spectacular, by Hooverphonic (downtempo, trip hop)\t0
Hunter Moon, in album Blood Year, by Russian Circles (post-rock)\t0
Arluck, in album Blood Year, by Russian Circles (math rock, post-rock)\t4
<end>
<start>
Пустите меня на танцпол, in album JANAVI, by HammAli & Navai\t0
Don't Start Now, in album Don't Start Now, by Dua Lipa (disco)\t3
<end>
<start>
I Like It, in album Invasion of Privacy, by Cardi B (rap)\t0
Say So, in album Hot Pink, by Doja Cat (pop)\t10
<end>
<start>
Esperar Pra Ver, in album Brazilian Compilation Series, Vol. 2, by Poolside\t0
Lucro (Descomprimindo), in album Duas Cidades, by BaianaSystem\t9
<end>
<start>
Future Nostalgia, in album Future Nostalgia, by Dua Lipa (pop)\t0
Some Say, in album Some Say, by Nea (pop)\t5
<end>''',\
"LastFM_8k":'''<start>
2 Wicky, in album A New Stereophonic Sound Spectacular, by Hooverphonic (downtempo, trip hop)\t0
Hunter Moon, in album Blood Year, by Russian Circles (post-rock)\t0
Arluck, in album Blood Year, by Russian Circles (math rock, post-rock)\t4
<end>
<start>
Пустите меня на танцпол, in album JANAVI, by HammAli & Navai\t0
Don't Start Now, in album Don't Start Now, by Dua Lipa (disco)\t3
<end>
<start>
I Like It, in album Invasion of Privacy, by Cardi B (rap)\t0
Say So, in album Hot Pink, by Doja Cat (pop)\t10
<end>
<start>
Esperar Pra Ver, in album Brazilian Compilation Series, Vol. 2, by Poolside\t0
Lucro (Descomprimindo), in album Duas Cidades, by BaianaSystem\t9
<end>
<start>
Future Nostalgia, in album Future Nostalgia, by Dua Lipa (pop)\t0
Some Say, in album Some Say, by Nea (pop)\t5
<end>'''
}

exp_example_exp_dict = {"Grocery_and_Gourmet_Food":'products that are fruit-based snacks, with a particular focus on cookies or bars that contain fruits such as strawberries, mixed fruits and coconut.',\
						# "ML_1MTOPK":"movies that have a dark, satirical, or ironic tone, particularly those that blend serious themes with a sense of humor or critique.",\
						"ML_1MTOPK":"movies with strong emotional or dramatic narratives, often featuring themes of romance, personal transformation, or significant character relationships, typically in genres such as drama, romance, or musicals.",\
						# "LastFM":"pop music with a catchy or upbeat style, particularly in tracks with a danceable or confident tone."
						# 'LastFM': "songs with upbeat, danceable, or pop-oriented vibes, often featuring elements of funk, disco, or tropical influences."
						"LastFM":"pop and dance-oriented music, with a particular focus on tracks that have strong, rhythmic beats or catchy melodies.",\
						"LastFM_8k":"pop and dance-oriented music, with a particular focus on tracks that have strong, rhythmic beats or catchy melodies."
						}

# prediction start here
pred_example_seq_dict = {
	"Grocery_and_Gourmet_Food":'''<start>
EDEN Millet, Whole Grain,16 -Ounce Pouches (Pack of 12)
Hodgon Mill Mulit Grain Milled Flaxseed &amp; Quinoa Hot Cereal (6x16oz)
Hodgson Mill Barley Bread Mix, 16-Ounce Units (Pack of 6)
Bob's Red Mill Soup Mix, 13 Bean, 29-Ounce Units (Pack of 4)
Fiber One Chewy Bars Oats &amp; Chocolate 36- 1.4 Oz Bars
Popchips 6-Flavor Variety Pack, 0.8-Ounce Single Serve Bags (Pack of 24)
Hodgson Mill Honey Whole Wheat Bread Mix, 16-Ounce Boxes (Pack of 6)
Popchips, Original, 3-Ounce Bags (Pack of 12)
Pure Bar Organic Chocolate Brownie, Gluten Free, Raw, Vegan,  1.7-Ounce Bars (Pack of 12)
<end>''',\
# 	"ML_1MTOPK": '''<start>
# Star Wars: Episode V - The Empire Strikes Back (1980)
# Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)
# Delicatessen (1991)
# Clockwork Orange, A (1971)
# Blade Runner (1982)
# Twelve Monkeys (1995)
# Terminator, The (1984)
# E.T. the Extra-Terrestrial (1982)
# Contact (1997)
# Sneakers (1992)
# Pi (1998)
# Usual Suspects, The (1995)
# Fargo (1996)
# Silence of the Lambs, The (1991)
# Manchurian Candidate, The (1962)
# Taxi Driver (1976)
# Nikita (La Femme Nikita) (1990)
# Fugitive, The (1993)
# Reservoir Dogs (1992)
# Seven (Se7en) (1995)
# Simple Plan, A (1998)
# <end>''',\
"ML_1MTOPK": ''''<start>
Clockwork Orange, A (1971) (Sci-Fi)
2001: A Space Odyssey (1968) (Drama, Mystery, Sci-Fi, Thriller)
Planet of the Apes (1968) (Action, Sci-Fi)
Thing, The (1982) (Action, Horror, Sci-Fi, Thriller)
Total Recall (1990) (Action, Adventure, Sci-Fi, Thriller)
Time Bandits (1981) (Adventure, Fantasy, Sci-Fi)
X-Men (2000) (Action, Sci-Fi)
Men in Black (1997) (Action, Adventure, Comedy, Sci-Fi)
Gladiator (2000) (Action, Drama)
Patriot, The (2000) (Action, Drama, War)
<end>''',\
"LastFM":'''<start>
I Like It, in album Invasion of Privacy, by Cardi B (rap)
Say So, in album Hot Pink, by Doja Cat (pop)
<end>''',\
"LastFM_8k":'''<start>
I Like It, in album Invasion of Privacy, by Cardi B (rap)
Say So, in album Hot Pink, by Doja Cat (pop)
<end>'''
}

# item_list = [5448, 373, 378, 2384, 4242]
# 			{meta_df.loc[4860,'title']}
pred_example_exp_dict = {
	"Grocery_and_Gourmet_Food": "products that are fruit-based snacks, with a particular focus on cookies or bars that contain fruits such as strawberries and mixed fruits.",\
	# "ML_1MTOPK": "movies that have a dark, satirical, or ironic tone, particularly those that blend serious themes with a sense of humor or critique." ,\
	"ML_1MTOPK":"movies with strong emotional or dramatic narratives, often featuring themes of romance, personal transformation, or significant character relationships, typically in genres such as drama, romance, or musicals.",\
	# "LastFM": 'pop music with a catchy or upbeat style, particularly in tracks with a danceable or confident tone.'
	# 'LastFM': "songs with upbeat, danceable, or pop-oriented vibes, often featuring elements of funk, disco, or tropical influences."
	"LastFM":"pop and dance-oriented music, with a particular focus on tracks that have strong, rhythmic beats or catchy melodies.",\
	"LastFM_8k":"pop and dance-oriented music, with a particular focus on tracks that have strong, rhythmic beats or catchy melodies."
}
pred_example_value_dict = {
	"Grocery_and_Gourmet_Food": "1",\
	"ML_1MTOPK":"6",
	'LastFM':"10",\
	'LastFM_8k':"10"
}

pred_example_lastItem_dict = {
	"Grocery_and_Gourmet_Food": "Pure Bar Organic Chocolate Brownie, Gluten Free, Raw, Vegan,  1.7-Ounce Bars (Pack of 12)",\
	"ML_1MTOPK":"Patriot, The (2000) (Action, Drama, War)",\
	'LastFM':"Say So, in album Hot Pink, by Doja Cat (pop)",\
	'LastFM_8k':"Say So, in album Hot Pink, by Doja Cat (pop)"
}

def getItemTitle(row, dataset):
	if dataset == "LastFM" or dataset == "LastFM_8k":
		if len(row['max_tag']) > 0:
			title = f"{row['title']}, in album {row['album_name']}, by {row['artist_name']} ({row['max_tag']})"
		else:
			title = f"{row['title']}, in album {row['album_name']}, by {row['artist_name']}"
	elif dataset == "ML_1MTOPK":
		category_list = row['genres'].split('|')
		categories = ', '.join(category_list)
		title = f"{row['title']} ({categories})"
	else:
		title = row['title']
	return title

if __name__ == "__main__":

	# method = "_recModel"
	# method = "_baseline"
	method = ""

	# base_model = "SASRec"
	# base_model = "BPRMF"
	# base_model = "TiMiRec"
	base_model = "LightGCN"

	sae_k = 16
	sae_scale_size = 16

	# dataset_name = "Grocery_and_Gourmet_Food"
	# dataset_name = "ML_1MTOPK"
	dataset_name = "LastFM_8k"

	SASRec_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"5e-05",16:"1e-05"},\
				"ML_1MTOPK":{4:"1e-05",8:"0.0001",16:"0.0001"},\
			   	"LastFM_8k":{4:"0.0001",8:"5e-05",16:"0.0001",32:"5e-05"}}
	SASRec_batch_dict = {"Grocery_and_Gourmet_Food":{4:4,8:4,16:4},\
				"ML_1MTOPK":{4:4,8:4,16:16},\
			   	"LastFM_8k":{4:4,8:4,16:4,32:4}}
	
	bprmf_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"1e-05",16:"5e-05"},\
				"ML_1MTOPK":{4:"0.0001",8:"0.0001",16:"0.0001"},\
			   	"LastFM_8k":{4:"5e-05",8:"1e-05",16:"1e-05",32:"1e-05"}}
	bprmf_batch_dict = {"Grocery_and_Gourmet_Food":{4:4,8:4,16:4},\
				"ML_1MTOPK":{4:16,8:16,16:16},\
			   	"LastFM_8k":{4:16,8:4,16:4,32:16}}
	
	DirectAU_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"5e-05",16:"1e-05"},\
				"ML_1MTOPK":{4:"0.0001",8:"5e-05",16:"5e-05"},\
			   	# "LastFM_8k":{4:"5e-05",8:"1e-05",16:"1e-05"}
				}
	DirectAU_batch_dict = {"Grocery_and_Gourmet_Food":{4:16,8:4,16:4},\
				"ML_1MTOPK":{4:4,8:16,16:16},\
			   	# "LastFM_8k":{4:16,8:4,16:4}
				}
	
	LightGCN_lr_dict = {"Grocery_and_Gourmet_Food":{4:"0.0001",8:"5e-05",16:"0.0001"},\
				"ML_1MTOPK":{8:"1e-05"},\
			   	"LastFM_8k":{4:"5e-05",8:"5e-05",16:"0.0001",32:"5e-05"}
				   }
	LightGCN_batch_dict = {"Grocery_and_Gourmet_Food":{4:4,8:4,16:16},\
				"ML_1MTOPK":{8:4},\
			   	"LastFM_8k":{4:4,8:16,16:4,32:16}
				}
	
	TiMiRec_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"1e-05",16:"0.0001"},\
				"ML_1MTOPK":{4:"0.0001",8:"1e-05",16:"5e-05"},\
			   	"LastFM_8k":{4:"0.0001",8:"0.0001",16:"0.0001",32:"0.0001"}
				   }
	TiMiRec_batch_dict = {"Grocery_and_Gourmet_Food":{4:16,8:4,16:4},\
				"ML_1MTOPK":{4:16,8:4,16:16},\
			   	"LastFM_8k":{4:16,8:16,16:16,32:16}
				}
	
	lr_dict = {"SASRec":SASRec_lr_dict,'BPRMF':bprmf_lr_dict,"LightGCN":LightGCN_lr_dict,"TiMiRec":TiMiRec_lr_dict,'DirectAU':DirectAU_lr_dict}
	batch_dict = {"SASRec":SASRec_batch_dict,"BPRMF":bprmf_batch_dict,"LightGCN":LightGCN_batch_dict,'TiMiRec':TiMiRec_batch_dict,'DirectAU':DirectAU_batch_dict}
	sae_lr = lr_dict[base_model][dataset_name][sae_k]
	batch_size = batch_dict[base_model][dataset_name][sae_k]


	train_test_data=[]
	result_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__trainTest_info_norm_haveTestAll{method}.jsonl"
	with open(result_file,'r') as f:
		all_data = f.readlines()
		for line in all_data:
			line = line.strip()
			data = json.loads(line)
			train_test_data.append(data)

	if dataset_name == "Grocery_and_Gourmet_Food":
		item_meta_file = "../../data/Grocery_and_Gourmet_Food/item_info.csv"
		item_name = "Grocery and Gourmet Food products"
		item_category = 'products'
	elif dataset_name =="ML_1MTOPK":
		item_meta_file = "../../data/MovieLens_1M/ML_1MTOPK/item_meta_all_info.csv"
		item_name = "movies"
		item_category = 'movies'
	elif dataset_name == 'LastFM':
		item_meta_file = "../../data/LastFM/item_meta_all_info.csv"
		item_name = "musics"
		item_category = 'musics'
	elif dataset_name == 'LastFM_8k':
		item_meta_file = "../../data/LastFM_8k/item_meta_all_info.csv"
		item_name = "musics"
		item_category = 'musics'
	meta_df = pd.read_csv(item_meta_file,delimiter='\t')
	meta_df.set_index('item_id',inplace=True)
	meta_df.rename({"track_name":'title'},inplace=True,axis=1)
	meta_df.fillna('',inplace=True)
	
	# result_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__prediction_norm_haveTestAll.csv"
	# print(result_file)
	# pred_df = pd.read_csv(result_file,delimiter='\t')
	if base_model == 'SASRec' or base_model == 'TiMiRec':
		file_path = f"../../log/{base_model}/{base_model}__{dataset_name}__0__lr=0/rec-{base_model}-test-history.csv"
	elif base_model == "BPRMF" or base_model == "LightGCN":
		file_path = f"../../log/{base_model}/{base_model}__{dataset_name}__0__lr=0/rec-{base_model}-test.csv"
	pred_df = pd.read_csv(file_path,delimiter="\t")
	# pred_df.set_index('index_id',inplace=True)
	for col in pred_df.columns.tolist():
		if pred_df [col].dtype == 'object':
			pred_df [col] = pred_df [col].apply(lambda v:eval(v))

	if base_model == "BPRMF" or base_model == "LightGCN":
		item_num = len(meta_df)
		item_range = range(1,item_num)
		number_range = range(1,9)
		length = len(pred_df)
		pred_df['history'] = [random.sample(item_range,random.choice(number_range)) for i in range(length)]


	example_item_sequence_train = exp_example_seq_dict[dataset_name]
	example_explanation_train = exp_example_exp_dict[dataset_name]
	example_latent_id = 10000

	test_example_item_sequence = pred_example_seq_dict[dataset_name]
	test_example_last_item = pred_example_lastItem_dict[dataset_name]
	test_example_value = pred_example_value_dict[dataset_name]
	test_example_explanation = pred_example_exp_dict[dataset_name]

	result_dict = dict()
	# latent_dict_file = f"../../log/SASRec_SAE/result_file/latents/SASRec_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__batch_size={batch_size}_latentDict_norm_v3.jsonl"
	# with open(latent_dict_file, 'r') as file:
	# 	result_dict = json.loads(file.readline())
		# file.write(json.dumps(result_dict))
	latent_info = dict()
	for data in train_test_data:
		for latent_id in data:
			print('[latent_id]',latent_id)
			content = data[latent_id]
			train_ids = content['train']
			if 'train_acts' in content:
				tag = 'train_acts'
			else:
				tag = 'train_act'
			train_acts = content[tag]
			act_strs= ""
			for index,id in enumerate(train_ids):
				history_list = pred_df.loc[id,'history']
				act_strs += "<start>\n"
				for item_id in history_list[-9:]:
					# if dataset_name == 'LastFM':
					# 	title = f"{meta_df.loc[item_id,'title']}, {meta_df.loc[item_id,'album_name']} by {meta_df.loc[item_id,'artist_name']}"
					# else:
					# 	title = meta_df.loc[item_id,'title']
					# import ipdb;ipdb.set_trace()
					title = getItemTitle(meta_df.loc[item_id], dataset_name)
					act_strs +=f"{title}\t0\n"
				# prediction_id = pred_df.loc[id,'rec_items'][0]
				# title = getItemTitle(meta_df.loc[prediction_id], dataset_name)
				# act_strs += f"{meta_df.loc[prediction_id,'title']}\t{train_acts[index]}\n<end>\n"
				# act_strs += f"{title}\t{train_acts[index]}\n"
				prediction_id = pred_df.loc[id,'rec_items'][0]
				title = getItemTitle(meta_df.loc[prediction_id], dataset_name)
				act_strs += f"{title}\t{train_acts[index]}\n<end>\n"
			
			instruct = [{'role':'system','content':f'''We're studying neurons in a recommendation model that is used to recommend {item_name}. Each neuron looks for some particular features in {item_name}. Look at the parts of the {item_category} the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.
The activation format is {item_category[:-1]}<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a not 0 activation value. The higher the activation value, the stronger the match.'''},\
			 	{'role':'user','content':f'''Neuron 10000
Activations:
{example_item_sequence_train}

Explanation of neuron 10000 behavior: the main thing this neuron does is find'''},\
				{'role':'assistant',"content":example_explanation_train},\
				{'role':'user','content':f'''Neuron {latent_id}
Activations:
{act_strs}

Explanation of neuron {latent_id} behavior: the main thing this neuron does is find'''}]
			# import ipdb;ipdb.set_trace()
			output = getLlamaResult(instruct)
			# import ipdb;ipdb.set_trace()
			# output = instruct
			result_dict[latent_id] = output
			print(output)
			# print(act_strs)


			if 'test' in content:
				test_ids = content['test'] + content['random']
				test_acts = content['test_acts'] + [0 for i in range(5)]
			else:
				test_ids = content['random']
				test_acts = [0 for i in range(5)]
			prediction_list = []
			for index,id in enumerate(test_ids):
				act_strs= ""
				history_list = pred_df.loc[id,'history']
				act_strs += "<start>\n"
				for item_id in history_list[-9:]:
					title = getItemTitle(meta_df.loc[item_id], dataset_name)
					act_strs +=f"{title}\n" # {meta_df.loc[item_id,'title']}
				prediction_id = pred_df.loc[id,'rec_items'][0]
				test_last_item = getItemTitle(meta_df.loc[prediction_id], dataset_name)# meta_df.loc[prediction_id,'title']
				
				act_strs += f"{test_last_item}\n<end>\n"

				test_prompt = [{'role':"system","content":f'''We're studying neurons in a recommendation model that is used to recommend {item_name}. Each neuron looks for some particular features in {item_name}. Look at an explanation of what the neuron does, and try to predict its activations on a particular {item_category[:-1]}.
The activation format is {item_category[:-1]}<tab>activation, and activations range from 0 to 10. Most activations will be 0.'''},\
# 			{'role':"user","content":f'''Neuron 0
# Explanation of neuron 0 behavior: the main thing this neuron does is find{example_explanation}'''},\
# 			{'role':"assistant","content":f'''Activations: 
# {example_item_sequence}'''},\
			{'role':"system","content":f"Now, we're going predict the activation of a new neuron on a single {item_category[:-1]}, following the same rules as the examples above. Activations still range from 0 to 10."},\
			{'role':"user","content":f'''Neuron {example_latent_id}
Explanation of neuron {example_latent_id} behavior: the main thing this neuron does is find {test_example_explanation}
Sequence:
{test_example_item_sequence}

Last {item_category[:-1]} in the sequence:
{test_example_last_item}

Last {item_category[:-1]} activation, considering the {item_category[:-1]} in the context in which it appeared in the sequence:'''},\
			
			{'role':'assistant',"content":test_example_value},\
			{"role":'user','content':f'''Neuron {latent_id}
Explanation of neuron {latent_id} behavior: the main thing this neuron does is find {result_dict[latent_id]}
Sequence:
{act_strs}

Last {item_category[:-1]} in the sequence:
{test_last_item}

Last {item_category[:-1]} activation, considering the {item_category[:-1]} in the context in which it appeared in the sequence:'''}]
				# import ipdb;ipdb.set_trace()
				output = getLlamaResult(test_prompt)
				# import ipdb;ipdb.set_trace()
				# output = test_prompt
				prediction_list.append(output)
				# print(act_strs)
				print(output,test_acts[index])
			if len(prediction_list)> 5:
				content['pred_test'] = prediction_list[:-5]
			content['pred_rand'] = prediction_list[-5:]
			latent_info[latent_id] = content
		# import ipdb;ipdb.set_trace()

	latent_dict_file = f"../../log/{base_model}_SAE/result_file/latents/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__latentDict_norm_allInfo_v5{method}.jsonl"
	print(latent_dict_file)
	with open(latent_dict_file, 'w') as file:
		file.write(json.dumps(result_dict))
	latent_dict_file = f"../../log/{base_model}_SAE/result_file/latents/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__latentDict_verification_norm_allInfo_v5{method}.jsonl"
	print(latent_dict_file)
	with open(latent_dict_file, 'w') as file:
		file.write(json.dumps(latent_info))
	# import ipdb;ipdb.set_trace()

