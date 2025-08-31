#!/usr/bin/env python3
"""
HelpSteer Integration Module
Implements core functionality of nVidia HelpSteer technology, including multi-dimensional evaluation and preference learning
"""

import json
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


class EvaluationDimension(Enum):
    """Evaluation dimension enumeration"""
    HELPFULNESS = "helpfulness"  # Helpfulness
    CORRECTNESS = "correctness"  # Correctness
    COHERENCE = "coherence"  # Coherence
    COMPLEXITY = "complexity"  # Complexity
    VERBOISITY = "verbosity"  # Verbosity


@dataclass
class HelpSteerResponse:
    """HelpSteer response data structure"""
    query: str
    response: str
    context: str
    scores: Dict[EvaluationDimension, int]
    overall_score: float
    metadata: Dict[str, Any]


class HelpSteerEvaluator:
    """HelpSteer evaluator for response assessment"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.evaluation_prompts = self._build_evaluation_prompts()
    
    def _build_evaluation_prompts(self) -> Dict[EvaluationDimension, PromptTemplate]:
        """Build evaluation prompts for each dimension"""
        prompts = {}
        
        # Helpfulness evaluation
        prompts[EvaluationDimension.HELPFULNESS] = PromptTemplate(
            input_variables=["query", "response", "context"],
            template="""
You are evaluating the helpfulness of an AI assistant's response to a user query.

Query: {query}
Context: {context}
Response: {response}

Rate the helpfulness of the response on a scale from 0 to 4, where:
0 = The response is not useful or helpful at all. The response completely missed the essence
of what the user wanted.

1 = The response is borderline unhelpful and mostly does not capture what the user was looking
for, but is still usable and helpful in a small way.

2 = The response is partially helpful but misses the overall goal of the user's query/input
in some way. The response did not fully satisfy what the user was looking for.

3 = The response is mostly helpful and mainly aligned with what the user was looking for,
but there is still some room for improvement.

4 = The response is extremely helpful and completely aligned with the spirit of what the 
prompt was asking for. 


Consider: 
- Does the response directly answer the user's question?
- Does it provide useful and actionable information?
- Does it go beyond the obvious to provide deeper insights?

Score (0-4):
"""
        )
        
        # Correctness evaluation
        prompts[EvaluationDimension.CORRECTNESS] = PromptTemplate(
            input_variables=["query", "response", "context"],
            template="""
You are evaluating the correctness of an AI assistant's response based on the provided context.

Query: {query}
Context: {context}
Response: {response}

Rate the correctness of the response on a scale from 0 to 4, where:
0 = The response is completely incorrect. All information provided is wrong, false or hallucinated.
If the prompt asks the assistant to do a task the task is not at all atempted, or the wrong task was attempted in the response.
The response is completely irrelevant to the prompt.

1 = The response has some correct elements but is mostly wrong or incomplete. The response may
contain multiple instances of hallucinations, false information, misleading information, or irrelevant
information. If the prompt asks the assistant to do a task, the task was attempted with a small amount of success.

2 = The response contains a mix of correct and incorrect information. The response may miss some details, contain misleading information, 
or minor hallucinations, but is more or less aligned with what the prompt asks for. If the prompt asks the assistant to perform a task, 
the task is attempted with moderate success but still has clear room for improvement. 

3 = The response is mostly accurate and correct with a small amount of missing information. It contains no misleading information or 
hallucinations. If the prompt asks the assistant to perform a task, the task is mostly successfully attempted.

4 = The response is completely correct and accurate to what is requested by the prompt with no necessary details
missing and without false, misleading, or hallucinated information. If the prompt asks the assistant to do a task, the 
task is completely done and addressed in the response.

Consider:
- Are the facts and claims in the response supported by the context?
- Are there any factual errors or misrepresentations?
- Does the response accurately interpret the source material?

Score (0-4):
"""
        )
        
        # Coherence evaluation
        prompts[EvaluationDimension.COHERENCE] = PromptTemplate(
            input_variables=["query", "response", "context"],
            template="""
You are evaluating the coherence of an AI assistant's response.

Query: {query}
Context: {context}
Response: {response}

Rate the coherence of the response on a scale from 0 to 4, where:
0 = The response is completely incomprehensible and no clear meaning or sensible message can be discerned from it.

1 = The response is mostly hard to follow, with inconsistencies, contradictions, confusing logic flow, or unclear
language used throughout, but here are some coherent/clear parts.

2 = There response is a little unclear. There are some inconsistencies or contradictions, run on sentences, confusing statements, 
or hard to follow sections of the response.

3 = The response is mostly clear and coherent, but there may be one or two places where the wording is confusing or the flow of the reponse
is a little hard to follow. Overall, the response can mostly be followed with a little room for improvement.

4 = The response is perfectly clear and slef-consistent throughout. There are no contradictory assertions or statements, the writing flows
logically, and following the train of thought/story is not challenging.

Consider:
- Is the response well-organized and logically structured?
- Is the language clear and accessible?
- Are complex concepts explained in understandable terms?

Score (0-4):
"""
        )
        
        # complexity evaluation
        prompts[EvaluationDimension.COMPLEXITY] = PromptTemplate(
            input_variables=["query", "response", "context"],
            template="""
You are evaluating the complexity of an AI assistant's response.

Query: {query}
Context: {context}
Response: {response}

Rate the complexity of the response on a scale from 0 to 4, where:
0 = The response uses very easy to understand language that is clear and completely interpretable by children, adults, and anyone
with a functional command of the language

1 = The response uses relatively straightforward language and wording, but some schooling through elementary
or a middle school in the language might be required to understand the response.

2 = People who have completed up through a high school education will probably be able to uderstand the vocabulary and sentence structure
used, but those at the basic level or children might struggle to understand the response.

3 = The response uses a fairly sophisticated vocabulary and terminology. Someone majoring in this subject at a college or university could have written it and 
would understand the response. An average adult who does not work or study in this area could not have written the response.

4 = An expert in the field or area could have written the response. It uses specific and technically relevant vocabulary. it contains elevated language that 
someone at the simple professional language of a lawyer, scientist, engineer, or doctor falls into this category.


Score (0-4):
"""
        )
        
        # verbosity evaluation
        prompts[EvaluationDimension.VERBOISITY] = PromptTemplate(
            input_variables=["query", "response", "context"],
            template="""
You are evaluating the verbosity of an AI assistant's response to the user query.

Query: {query}
Context: {context}
Response: {response}

Rate the verbosity of the response on a scale from 0 to 4, where:
0 = The response is short, to the point, and the most concise it cna be. No additional information is provided outside of
what is requested by the prompt.

1 = The response is on the shorter side but could still have words, details, and/or text removed before it's at a bare minimum of 
what the response is trying to convey.

2 = The response isn't especially long or short given what the prompt is asking of the model. The length is adequate for conveying a full response but isn't 
particularly wordy nor particularly concise.

3 = The response is on the longer side but could still have more added to it before it is considered fully detailed or rambling.

4 = The response is particularly lengthy, wordy, and/or extensive with extra details given what the prompt requested from the assistant model. The response can 
be verbose regardless of if the length is due to repetition and incoherency or if it is due to rich and insightful detail.



Score (0-4):
"""
        )
        
        return prompts
    
    def evaluate_response(self, query: str, response: str, context: str) -> HelpSteerResponse:
        """Evaluate a single response"""
        scores = {}
        
        # Evaluate each dimension
        for dimension in EvaluationDimension:
            prompt = self.evaluation_prompts[dimension]
            chain = LLMChain(llm=self.llm_client.llm, prompt=prompt)
            
            try:
                score_str = chain.run(query=query, response=response, context=context).strip()
                # Extract numeric score
                score = self._extract_score(score_str)
                scores[dimension] = score
            except Exception as e:
                print(f"Error evaluating dimension {dimension.value}: {e}")
                scores[dimension] = 2  # Default medium score
        
        # Calculate overall score (weighted average)
        overall_score = self._calculate_overall_score(scores)
        
        return HelpSteerResponse(
            query=query,
            response=response,
            context=context,
            scores=scores,
            overall_score=overall_score,
            metadata={
                "evaluation_method": "helpsteer",
                "dimensions_evaluated": [dim.value for dim in scores.keys()]
            }
        )
    
    def _extract_score(self, score_str: str) -> float:
        """Extract score from evaluation result"""
        try:
            # Try to parse number directly
            score = float(score_str)
            return max(0, min(4, score))  # Limit to 0-4 range
        except ValueError:
            # If direct parsing fails, try to extract number from text
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', score_str)
            if numbers:
                score = float(numbers[0])
                return max(0, min(4, score))
            return 2  # Default score
    
    def _calculate_overall_score(self, scores: Dict[EvaluationDimension, float]) -> float:
        """Calculate overall score"""
        # Can set different weights for different dimensions
        weights = {
            EvaluationDimension.HELPFULNESS: 0.1,
            EvaluationDimension.CORRECTNESS: 0.7,
            EvaluationDimension.COHERENCE: 0.2,
            EvaluationDimension.COMPLEXITY: 0.0,
            EvaluationDimension.VERBOISITY: -0.15
        }
        
        weighted_sum = sum(scores[dim] * weights[dim] for dim in scores.keys())
        return round(weighted_sum, 2)


class HelpSteerTrainer:
    """HelpSteer trainer"""
    
    def __init__(self, llm_client, evaluator: HelpSteerEvaluator):
        self.llm_client = llm_client
        self.evaluator = evaluator
        self.preference_prompt = self._build_preference_prompt()
    
    def _build_preference_prompt(self) -> PromptTemplate:
        """Build preference learning prompt"""
        return PromptTemplate(
            input_variables=["query", "context", "better_response", "worse_response"],
            template="""
You are learning from human preferences to improve AI assistant responses.

Given a user query and two possible responses, analyze why the better response is preferred over the worse one.

Query: {query}
Context: {context}

Better Response: {better_response}
Worse Response: {worse_response}

Please analyze the differences and explain why the better response is preferred. Focus on:
1. Helpfulness: Which response better addresses the user's needs?
2. Correctness: Which response is more accurate based on the context?
3. Clarity: Which response is easier to understand?
4. Conciseness: Which response is more efficient?
5. Relevance: Which response stays more focused on the query?

Analysis:
"""
        )
    
    def generate_preference_data(self, query: str, context: str, 
                               response_a: str, response_b: str) -> Dict[str, Any]:
        """Generate preference data"""
        # Evaluate two responses
        eval_a = self.evaluator.evaluate_response(query, response_a, context)
        eval_b = self.evaluator.evaluate_response(query, response_b, context)
        
        # Determine which is better
        if eval_a.overall_score > eval_b.overall_score:
            better_response = response_a
            worse_response = response_b
            better_score = eval_a.overall_score
            worse_score = eval_b.overall_score
        else:
            better_response = response_b
            worse_response = response_a
            better_score = eval_b.overall_score
            worse_score = eval_a.overall_score
        
        # Generate preference analysis
        chain = LLMChain(llm=self.llm_client.llm, prompt=self.preference_prompt)
        analysis = chain.run(
            query=query,
            context=context,
            better_response=better_response,
            worse_response=worse_response
        )
        
        return {
            "query": query,
            "context": context,
            "better_response": better_response,
            "worse_response": worse_response,
            "better_score": better_score,
            "worse_score": worse_score,
            "score_difference": better_score - worse_score,
            "analysis": analysis,
            "evaluations": {
                "better": eval_a if eval_a.overall_score > eval_b.overall_score else eval_b,
                "worse": eval_b if eval_a.overall_score > eval_b.overall_score else eval_a
            }
        }
    
    def save_preference_data(self, data: Dict[str, Any], filepath: str):
        """Save preference data"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_preference_data(self, filepath: str) -> Dict[str, Any]:
        """Load preference data"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


class HelpSteerPromptOptimizer:
    """HelpSteer prompt optimizer"""
    
    def __init__(self, llm_client, evaluator: HelpSteerEvaluator):
        self.llm_client = llm_client
        self.evaluator = evaluator
        self.optimization_prompt = self._build_optimization_prompt()
    
    def _build_optimization_prompt(self) -> PromptTemplate:
        """Build prompt optimization prompt"""
        return PromptTemplate(
            input_variables=["query", "context", "current_response", "evaluation", "target_improvements"],
            template="""
You are optimizing an AI assistant's response based on evaluation feedback.

Query: {query}
Context: {context}
Current Response: {current_response}

Evaluation Scores:
{evaluation}

Target Improvements: {target_improvements}

Please provide an improved version of the response that addresses the identified weaknesses.
Focus on the specific areas that need improvement while maintaining the strengths.

Improved Response:
"""
        )
    
    def optimize_response(self, query: str, context: str, current_response: str,
                         target_improvements: List[str]) -> str:
        """Optimize response"""
        # Evaluate current response
        evaluation = self.evaluator.evaluate_response(query, current_response, context)
        
        # Format evaluation results
        eval_text = "\n".join([
            f"- {dim.value}: {score}/4" 
            for dim, score in evaluation.scores.items()
        ])
        
        # Generate optimization prompt
        chain = LLMChain(llm=self.llm_client.llm, prompt=self.optimization_prompt)
        improved_response = chain.run(
            query=query,
            context=context,
            current_response=current_response,
            evaluation=eval_text,
            target_improvements=", ".join(target_improvements)
        )
        
        return improved_response.strip()


class HelpSteerSystem:
    """Complete HelpSteer system"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.evaluator = HelpSteerEvaluator(llm_client)
        self.trainer = HelpSteerTrainer(llm_client, self.evaluator)
        self.optimizer = HelpSteerPromptOptimizer(llm_client, self.evaluator)
    
    def evaluate_and_improve(self, query: str, context: str, response: str,
                           improvement_targets: List[str] = None) -> Dict[str, Any]:
        """Evaluate and improve response"""
        # Evaluate current response
        evaluation = self.evaluator.evaluate_response(query, response, context)
        
        result = {
            "original_response": response,
            "evaluation": evaluation,
            "improved_response": None,
            "improvement_analysis": None
        }
        
        # If improvement is needed
        if improvement_targets:
            improved_response = self.optimizer.optimize_response(
                query, context, response, improvement_targets
            )
            
            # Evaluate improved response
            improved_evaluation = self.evaluator.evaluate_response(
                query, improved_response, context
            )
            
            result["improved_response"] = improved_response
            result["improved_evaluation"] = improved_evaluation
            result["improvement_analysis"] = {
                "score_improvement": improved_evaluation.overall_score - evaluation.overall_score,
                "dimension_improvements": {
                    dim.value: improved_evaluation.scores[dim] - evaluation.scores[dim]
                    for dim in evaluation.scores.keys()
                }
            }
        
        return result
    
    def generate_training_data(self, queries: List[str], contexts: List[str],
                             responses_a: List[str], responses_b: List[str],
                             output_file: str):
        """Generate training data"""
        training_data = []
        
        for i, (query, context, resp_a, resp_b) in enumerate(
            zip(queries, contexts, responses_a, responses_b)
        ):
            print(f"Processing sample {i+1}/{len(queries)}...")
            
            preference_data = self.trainer.generate_preference_data(
                query, context, resp_a, resp_b
            )
            training_data.append(preference_data)
        
        # Save training data
        self.trainer.save_preference_data(training_data, output_file)
        print(f"Training data saved to: {output_file}")
        
        return training_data 