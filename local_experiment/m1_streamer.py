from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from time import perf_counter
import asyncio


checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

short_token_inputs = [

    "How many helicopters can a human eat in one sitting?",

 ]


long_token_inputs = ["""Sometimes, what people truly need is not a completely new beginning, but simply some time to reorganize their lives. Life rarely unfolds in the order we originally imagine. We may think we will spend an entire year focusing on work, only to change jobs unexpectedly. We may assume that we will continue living in the same place, only to move because of family, immigration, career opportunities, or other circumstances. We may plan to exercise every day, read more books, and learn new skills, only to discover that simply completing everything we need to do each day already consumes most of our energy.

Living in a fast-paced city can create the illusion that everyone else is constantly moving forward. Every morning, commuters rush through train stations, people open their laptops in coffee shops and begin working, office buildings remain brightly lit late into the night, and social media is filled with promotions, expensive vacations, new homes, investments, and other milestones. When we see these things, it is easy to wonder whether we are moving too slowly. Yet real life has never been a race in which everyone starts from the same position and competes to reach the same finish line. Everyone has different family circumstances, responsibilities, financial pressures, priorities, and stages of life. Because of that, comparing the apparent speed of one person's progress with another's rarely tells us much.

Some periods of life are meant for rapid growth, while others are meant for simply maintaining stability. Maintaining stability is not the same as falling behind. Taking care of family, keeping daily life organized, handling necessary administrative tasks, getting enough sleep, or simply having a little quiet time to think may not look productive from the outside, but they are still meaningful parts of life. As responsibilities increase, the free time that once seemed unlimited can disappear surprisingly quickly. In the past, one evening might have been enough to finish an entire course, watch a movie, play games for several hours, or work on a personal project. Later, the same evening may need to be divided into many small pieces. This does not necessarily mean that a person has become less capable. It may simply mean that the conditions of their life have changed.

One of the hardest things is that we often use our past selves as the standard for judging our present selves. If we were once able to work eight hours a day and study for another three hours at night, we may assume that we should still be able to maintain the same level of output. But if our responsibilities have changed, our standards should change as well. Maturity does not mean maintaining the same level of productivity forever. It means understanding when to push forward and when to preserve energy.

Reorganizing life does not necessarily require a grand plan. It can begin with very simple actions: getting rid of things that have not been used for years, turning off unnecessary notifications, writing down the tasks that genuinely matter, and identifying the areas that deserve attention over the next few months. Much of our anxiety does not come from having too many things to do, but from allowing every task, concern, and possibility to exist in our minds at the same time. Once everything is written down and separated into priorities, we often realize that some things can wait, some things do not need to be done at all, and some things only seem important because other people are doing them.

Some of the most valuable decisions in life are not about what we should add, but about what we should stop doing. We can stop consuming information that adds no value, stop comparing ourselves with people whose circumstances are completely different, stop accepting opportunities simply to prove that we are successful, and stop treating every moment of rest as laziness. Once we understand what does not deserve our attention, life can become surprisingly lighter.

This does not mean giving up on progress. In fact, sustainable progress usually comes from consistent accumulation rather than short bursts of effort. Reading a few pages every day, completing one small project each week, regularly learning a useful skill, exercising occasionally, and spending meaningful time with family may seem insignificant when considered individually. But after a year, these small actions can produce a completely different outcome. Many changes are difficult to notice while they are happening. It is only when we look back that we realize how far we have traveled.

So, if life currently feels somewhat uncertain or disorganized, there may be no need to immediately find a perfect answer. It can be enough to take care of the things that genuinely matter right now and preserve the ability to make choices about the future. Direction is not always something we discover by sitting alone and thinking about it. Sometimes it emerges gradually through living, experimenting, making mistakes, learning, and adjusting.

Perhaps the most important thing in life is not maintaining the fastest possible speed, but knowing why we are moving forward and recognizing when it is worth stopping to appreciate the people and circumstances around us. Once we stop worrying so much about proving that we are not falling behind, we create enough space to ask a more meaningful question: if we did not have to explain our choices to anyone else, where would we truly want to spend our time? That question may be more valuable than knowing exactly what we should do next.
"""]


def token_calculate_time(text, title):

    ttft_list = []
    tpot_list = []
    
    for k in range(3):

        inputs = tokenizer(text, padding=True, padding_side="left", truncation=True, return_tensors="pt")
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=100, do_sample=False)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        

        start = perf_counter()
        thread.start()
        generated_text = ""

        has_first_token = False

        for new_text in streamer:

            if new_text and not has_first_token:
                has_first_token = True
                first_token_time = perf_counter()

            generated_text += new_text

        end = perf_counter()

        print(generated_text)

        token_num = len(tokenizer.encode(generated_text))

        ttft_list.append(first_token_time - start)
        tpot_list.append((end - first_token_time) / (token_num - 1))


    print(f"--- {title} ---")    
    print(f"Input tokens: {tokenizer.encode(text)['input_ids'].shape[1]}")}")
    print(f"TTFT: {ttft_list.sort()[1]} seconds")
    # print(f"Total tokens generated: {token_num}")
    print(f"TPOT: {tpot_list.sort()[1]} seconds per token")


raw_inputs = ["How are you today?"]

inputs = tokenizer(raw_inputs, padding=True, padding_side="left", truncation=True, return_tensors="pt")
generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)

token_calculate_time(short_token_inputs, "Short token input")
token_calculate_time(long_token_inputs, "Long token input")