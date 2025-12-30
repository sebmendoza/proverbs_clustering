import os
from typing import List, Dict, Optional
from anthropic import Anthropic
from dotenv import load_dotenv
from cluster_builder import ClusterBuilder

load_dotenv()


class BatchRequester:
    def __init__(self, api_key: Optional[str] = None):
        self.client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.current_batch_id = None
        self.model = "claude-sonnet-4-5-20250929"
        self.max_tokens = 50
        self.system_prompt = "You are a Biblical Scholar and Theologian. You provide 4 word responses only."

    def _build_prompt(self, verses: List[str]) -> str:
        verses_text = "\n".join(verses)
        prompt = (
            "Here are verses from the Book of Proverbs that were grouped together "
            "by semantic similarity. Provide a 4-word thematic summary that captures "
            "what these verses have in common. Just the 4 words, nothing else.\n\n"
            f"{verses_text}"
        )
        return prompt

    def _build_single_request(self, custom_id: str, verses: List[str]) -> Dict:
        prompt = self._build_prompt(verses)
        return {
            "custom_id": custom_id,
            "params": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": self.system_prompt,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            }
        }

    def build_batch_requests(self, clusters: List[Dict]) -> List[Dict]:
        all_requests = []
        for cluster in clusters:
            custom_id = f"cluster_{cluster['cluster_id']}"
            request = self._build_single_request(custom_id, cluster['verses'])
            all_requests.append(request)

        print(f"Built {len(all_requests)} batch requests")
        return all_requests

    def submit_batch(self, requests: List[Dict]) -> str:
        batch_response = self.client.messages.batches.create(requests=requests)
        self.current_batch_id = batch_response.id

        print(f"Batch submitted: {batch_response.id}")
        print(f"Status: {batch_response.processing_status}")
        print(f"Total requests: {len(requests)}")

        with open("cluster_batch_id.txt", "a") as f:
            f.write("Batch for " + str(len(requests)) +
                    " requests: " + batch_response.id + "\n")
            print(f"Batch ID saved to cluster_batch_id.txt")

        return batch_response.id

    def retrieve_batch(self, batch_id: str) -> str:
        batch_response = self.client.messages.batches.retrieve(batch_id)
        if batch_response.processing_status != "ended":
            print(
                f"Batch is still {batch_response.processing_status}. Please wait.")
            return []

        # Get results
        results = self.client.messages.batches.results(batch_id)
        for result in results:
            print(result.result.message.content[0].text)

        return batch_response


def run_full_workflow(json_filepath: str):
    print("=== Step 1: Loading Clusters ===")
    cluster_builder = ClusterBuilder(json_filepath)
    clusters = cluster_builder.get_clusters()
    print("\n=== Step 2: Submitting Batch ===")
    batch_requester = BatchRequester()
    requests = batch_requester.build_batch_requests(clusters)
    print(requests)
    batch_id = batch_requester.submit_batch(requests)
    print(f"\n✓ Batch submitted successfully!", batch_id)


def run_retrieval(ids: List[str]):
    for id in ids:
        batch_requester = BatchRequester()
        res = batch_requester.retrieve_batch(id)


if __name__ == "__main__":
    json_file = "experiments/experiment_20251223_114054/k_20/clusters.json"
    # run_full_workflow(json_file)
    ids = ["msgbatch_01ANJMDLRpx1GJV6dj5PVU7s"]
    run_retrieval(ids)
