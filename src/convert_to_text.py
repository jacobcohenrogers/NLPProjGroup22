import os
import re

# Replace this with your actual essay string
full_text = """
1. Your first day at college
The morning air was sticky with late-summer humidity, and my cotton shirt clung uncomfortably to my back as I lugged a duffel bag across the lawn. I remember the scent of freshly mowed grass mixed with nerves and possibility. Everything about campus felt oversized—arched brick buildings, winding paths, voices tumbling from every direction. I found my dorm by accident, tripping over a curb while trying to read the map upside down.

Later, after my parents drove away with teary waves, I sat on the edge of my new twin bed and listened to the unfamiliar chorus outside: the roll of skateboard wheels, someone playing guitar off-key, laughter echoing through the hall. I felt both invisible and on display, like a tourist in my own life. But then my roommate offered to split a mango with me—we peeled it over a paper towel, juice sticky on our fingers, talking about music and books. And just like that, it started to feel less like a threshold and more like a beginning.

2. Who is your hero, and why?
My abuela, Lourdes, is my quiet hero. She never learned to read fluently, but she can recite entire recipes from memory and navigate the chaos of a family kitchen with the grace of a conductor. Every morning, she wakes before the sun and brews café con leche in her tiny stove-top pot, humming boleros under her breath. There’s a kind of magic in how she can thread a needle in dim light or tell stories that make time slow down.

She taught me that strength isn’t always loud. It can be the way you endure, the way you show up with hands ready and heart open. When I was twelve and heartbroken over a failed spelling bee, she said, “Mija, you don’t need to know every word—you just need to use the right ones when it matters.” I think about that often when I write.

3. A time you overcame a challenge
I once got hopelessly lost in Kyoto. I’d wandered away from the tourist-friendly streets, enchanted by a quiet alley lined with tiny lanterns and the faint smell of sesame oil. Then my phone died. No maps, no translation app, no clue how to ask for help. Panic bloomed like a too-fast heartbeat, and I stood frozen in front of a shop selling handmade fans.

But then the shopkeeper, an elderly woman with silver hair in a bun, noticed my face and beckoned me inside. She poured me barley tea and pulled out a paper map from beneath the counter. We didn’t share a language, but she drew little symbols, pointed, mimed walking. When I stepped back outside, I wasn’t just oriented—I was grounded again. That moment reminded me that kindness is its own kind of compass, and sometimes, the challenge isn’t finding your way, but trusting someone to help you get there.

4. What does success mean to you?
For a long time, I thought success meant awards, publications, applause—the things that look good in frames or sound impressive at dinner parties. But lately, it’s taken on a quieter shape. It’s the feeling of standing on a remote hilltop in Patagonia, my fingers numb from wind but my lungs full of air so clean it almost hurt. It’s finishing a draft of a blog post and knowing I said something true, even if only five people read it.

Success, to me now, is being deeply present. It's holding eye contact during an honest conversation. It’s cooking arroz con pollo just like my mom, and hearing my little cousin say, “Tastes like hers.” It’s showing up for my friends. If I can live a life stitched together with those small, golden threads, I’ll feel like I’ve made it.

5. A memorable journey
I once rode a bus across the Balkans in winter. The windows fogged up within minutes, so I had to press my forehead to the glass to glimpse the snowy fields outside. There was a woman beside me knitting a red scarf, and the soft clicking of her needles became the rhythm of the ride. At every stop, locals would board with bags full of bread or cheese wrapped in cloth, the scent mixing with diesel and damp wool.

What made the journey memorable wasn’t the destination—it was the stillness between places. The way people leaned into sleep, heads bumping against windows. The sound of radio static as we passed through mountain tunnels. I realized then how much beauty there is in waiting, in transit. We’re so often chasing arrivals, but sometimes, the best part is the gentle in-between.

6. A lesson learned from failure
During my first semester of grad school, I tried to write a paper that I thought would impress my professor. I used elaborate metaphors, cited obscure theorists, and buried my real thoughts under layers of academic fog. When I got it back, there was a single comment: “You’re hiding.”

At first, I was crushed. I’d spent days crafting that piece. But as I reread it, I realized he was right. I wasn’t writing what I believed—I was performing. That moment changed everything. I started writing like I was talking to someone I cared about. I stopped chasing approval and started seeking connection. Failing that paper taught me that clarity and honesty are braver than brilliance.

7. Your favorite hobby and why you love it
Photography snuck into my life through the back door. I started with phone snapshots on hikes, mostly to remember where I’d been. But then I started noticing how light filtered through trees at golden hour or how street cats in Morocco always find the same patch of sun. Now I carry my camera like a second pair of eyes.

What I love most is how photography trains you to see. You start noticing everything—the curve of a stranger’s smile, the way rain gathers in a flower’s petals, the quiet choreography of daily life. When I’m behind the lens, I feel tethered to the world in a softer, more intentional way. It’s a kind of prayer, a way of saying, “Look. This matters.”

8. A conflict you resolved
My best friend Clara and I once had a falling out over something small—a misread text, a forgotten plan, the kind of thing that festers when you’re both tired and trying too hard. We went weeks without talking, each convinced the other didn’t care. But one rainy afternoon, I passed the café where we used to go after class and saw her through the window, staring into a mug with that same pensive look I knew so well.

I went in. We didn’t say sorry right away. We just ordered chai and sat in silence until the awkwardness melted. Then, slowly, we unpacked the missteps. I learned how important it is to name things out loud before they harden into silence. Conflict resolution, I’ve found, isn’t about being right—it’s about being willing.

9. A significant childhood memory
When I was seven, I built a fort in my abuela’s backyard out of beach towels and chairs. I called it my “reading cave” and brought in a flashlight, a notebook, and a half-eaten pack of cookies. The afternoon sun made the orange towel glow like stained glass. I remember the crinkle of pages as I turned them, the hum of bees somewhere nearby, the faint smell of hibiscus.

That little fort was my first sanctuary—a space where stories and dreams mingled freely. I think about it often when I’m overwhelmed. That small act of carving out space for myself taught me something I still carry: even in chaos, you can create a corner of calm. A place to breathe, to imagine, to be.

10. Your hopes for the future
I don’t need a mansion or a five-year plan. I just hope for a life with rhythm—a slow, steady beat of curiosity, love, and purpose. I want to wake up in places that surprise me, eat meals that remind me of home, write things that make people feel less alone. I want to keep asking questions, keep sitting on café balconies with a notebook and a half-drunk coffee, watching the world tilt and turn.

And one day, maybe, I’ll settle down near the sea. I’ll have a little garden with too much mint, a crooked bookshelf, and a window that catches the morning light just right. I’ll tell stories to anyone who’ll listen—stories about the world, about people, about the small moments that make a life. And that, I think, would be enough.
"""

# Output directory
output_dir = os.path.join("..", "data", "raw_text", "training", "Gomez")
os.makedirs(output_dir, exist_ok=True)

# Split based on numbered headers like: 1. Title
split_pattern = r"\n(?=\d+\.\s)"
parts = re.split(split_pattern, full_text.strip())

# Each part will still have a "1. Title" header. Remove it and save the rest.
for i, part in enumerate(parts):
    lines = part.strip().split("\n", 1)
    if len(lines) > 1:
        content = lines[1].strip()  # skip the "1. Title" line
    else:
        content = ""  # fallback in case content is missing

    output_path = os.path.join(output_dir, f"gomez_{i}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
