import sqlite3
import random

SQLITE_DB='/Users/mansi/projects/langchain/db/prompts.db'

def get_all_prompts():
    connection = sqlite3.connect(SQLITE_DB)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM prompts")
    prompts = cursor.fetchall()
    connection.close()
    return prompts

# Example usage to retrieve all prompts from the database

def generate_prompt():
    all_prompts = get_all_prompts()
    prompt_num = random.randint(0, len(all_prompts)-1)
    #for prompt in all_prompts:
    #    print(prompt[1])
    return  all_prompts[prompt_num][1]