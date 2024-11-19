import sys

sys.path.append("..")

from prompts.prompt_manager import PromptManager

support_prompt = PromptManager.get_prompt(
    "ticket_analysis", pipeline="support", ticket={}
)

helpdesk_prompt = PromptManager.get_prompt(
    "ticket_analysis", pipeline="helpdesk", ticket={}
)

print(support_prompt)
print(helpdesk_prompt)
