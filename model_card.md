# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **VibeFinder 1.0**  

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

---

## 3. How the Model Works  

Think of the recommender as a judge at a talent show. It looks at every song in the catalog, one at a time, gives each one a score out of 5 based on how well it fits what you like, and then shows you the five highest-scoring songs.

**What it knows about each song.** Every song has a genre (like lofi or rock), a mood (like chill or intense), an energy level from 0 (very calm) to 1 (very intense), and a valence level from 0 (sad or dark) to 1 (happy or bright). The catalog also lists tempo, danceability, acousticness and other details, but the scoring ignores those. Fast songs are usually high-energy songs, so counting tempo too would mostly reward the same thing twice.

**What it knows about you.** Your taste profile is your favorite genre, your favorite mood, how energetic you want the music to be, and optionally how upbeat you want it to be. If you don't say how upbeat, it assumes you're in the middle.

**How a song earns points.**

- **Genre (up to 2 points).** A song in your favorite genre gets 2 points. A song in a closely related genre, like "indie pop" when you like "pop", gets 1 point. Genre is worth the most because it's the most stable part of someone's taste: a metal fan rarely wants a lofi song, even on a calm day.
- **Mood (1 point).** A song with exactly your mood gets 1 point. Mood is worth less than genre because it changes from day to day, and because the energy and valence scores already reflect it.
- **Energy (up to 1.5 points).** Rather than "more energy is better", the closer a song's energy is to the energy you asked for, the more points it gets. A perfect match earns all 1.5 points, a song that's a little off earns a bit less, and a song at the opposite extreme earns nothing.
- **Valence (up to 0.5 points).** This works the same way as energy but counts for less. It mostly acts as a tiebreaker between songs that are otherwise very similar.

Add those up and the best possible song scores 5. Energy alone can never beat a genre match (1.5 is less than 2), so the right genre usually wins. Every time a song earns points, the system writes down why (for example, "mood match +1.0"), so each recommendation comes with a short explanation.

**What I changed from the starter.** The starter code didn't score anything: it loaded no songs and returned no recommendations, and the class version simply handed back the first few songs in the file. I added the point system above, partial credit for related genres, and the valence tiebreaker, and I added the "why" explanations so you can see exactly how each score was earned.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran  

No need for numeric metrics unless you created some.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  
