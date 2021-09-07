import logging
import os
import re
from functools import lru_cache
from urllib.parse import unquote
import pandas as pd
import streamlit as st
# import wikipedia
from codetiming import Timer
from fuzzysearch import find_near_matches
# from googleapi import google
from transformers import AutoTokenizer, pipeline

from annotator import annotated_text
from preprocess import ArabertPreprocessor

logger = logging.getLogger(__name__)

# wikipedia.set_lang("ar")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

preprocessor = ArabertPreprocessor("wissamantoun/araelectra-base-artydiqa")
logger.info("Loading Pipeline...")
tokenizer = AutoTokenizer.from_pretrained("wissamantoun/araelectra-base-artydiqa")
qa_pipe = pipeline("question-answering", model="wissamantoun/araelectra-base-artydiqa")
logger.info("Finished loading Pipeline...")


@lru_cache(maxsize=100)
def get_results(question):
    logger.info("\n~~~~")
    logger.info(f"Question: {question}")

    df = pd.read_csv('ar_dataset_therapy.csv')
    full_len_sections = df['answerText'].tolist()

    logger.info(f"(Pre QA Pipe) Total Full Sections: {len(full_len_sections)}")
 
    reader_time = Timer("electra", text="Reader Time: {:.2f}", logger=logging.info)
    reader_time.start()
    results = qa_pipe(
        question=[preprocessor.preprocess(question)] * len(full_len_sections),
        context=[preprocessor.preprocess(x) for x in full_len_sections],
    )

    if not isinstance(results, list):
        results = [results]
        
    logger.info(f"Total Full Sections: {len(full_len_sections)}")

    for result, section in zip(results, full_len_sections):
        result["original"] = section
        answer_match = find_near_matches(
            " " + preprocessor.unpreprocess(result["answer"]) + " ",
            result["original"],
            max_l_dist=min(5, len(preprocessor.unpreprocess(result["answer"])) // 2),
            max_deletions=0,
        )
        try:
            result["new_start"] = answer_match[0].start
            result["new_end"] = answer_match[0].end
            result["new_answer"] = answer_match[0].matched
            result["link"] = "!#"
        except:
            result["new_start"] = result["start"]
            result["new_end"] = result["end"]
            result["new_answer"] = result["answer"]
            result["original"] = preprocessor.preprocess(result["original"])
            result["link"] = "!#"
        logger.info(f"Answers: {preprocessor.preprocess(result['new_answer'])}")

    sorted_results = sorted(results, reverse=True, key=lambda x: x["score"])

    return_dict = {}
    return_dict["title"] = "MentalHealth Dataset"
    return_dict["results"] = sorted_results

    reader_time.stop()
    logger.info(f"Total time spent: {reader_time.last}")
    return return_dict


def shorten_text(text, n, reverse=False):
    if text.isspace() or text == "":
        return text
    if reverse:
        text = text[::-1]
    words = iter(text.split())
    lines, current = [], next(words)
    for word in words:
        if len(current) + 1 + len(word) > n:
            break
            lines.append(current)
            current = word
        else:
            current += " " + word
    lines.append(current)
    if reverse:
        return lines[0][::-1]
    return lines[0]


def annotate_answer(result):
    annotated_text(
        shorten_text(
            result["original"][: result["new_start"]],
            500,
            reverse=True,
        ),
        (result["new_answer"], "جواب", "#8ef"),
        shorten_text(result["original"][result["new_end"] :], 500) + " ...... إلخ",
    )


if __name__ == "__main__":
    results_dict = get_results("ما هو نظام لبنان؟")
    for result in results_dict["results"]:
        annotate_answer(result)
        f"[**المصدر**](<{result['link']}>)"
