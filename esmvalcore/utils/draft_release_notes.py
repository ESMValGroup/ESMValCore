"""Draft release notes.

To use this tool, follow these steps:
1) `pip install pygithub`
2) Create an access token and store it in the file ~/.github_api_key, see:
https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line
3) set PREVIOUS_RELEASE to the date/time of the previous release in the code below

"""
import datetime
from pathlib import Path

try:
    from github import Github
except ImportError:
    print("Please `pip install pygithub`")

GITHUB_API_KEY = Path("~/.github_api_key").expanduser().read_text().strip()
GITHUB_REPO = "ESMValGroup/ESMValCore"


def draft_notes_since(previous_release_date, labels):
    """Draft release notes containing the merged pull requests.

    Arguments
    ---------
    previous_release_date: datetime.datetime
        date of the previous release
    labels: list
        list of GitHub labels that deserve separate sections
    """
    session = Github(GITHUB_API_KEY)
    repo = session.get_repo(GITHUB_REPO)
    pulls = repo.get_pulls(
        state='closed',
        sort='updated',
        direction='desc',
    )

    lines = {}
    for pull in pulls:
        if pull.merged:
            if pull.closed_at < previous_release_date:
                break
            pr_labels = {label.name for label in pull.labels}
            for label in labels:
                if label in pr_labels:
                    break
            else:
                label = 'enhancement'

            user = pull.user
            username = user.login if user.name is None else user.name
            line = (f"- {pull.title} (#{pull.number}) "
                    f"[{username}](https://github.com/{user.login})")
            if label not in lines:
                lines[label] = []
            lines[label].append((pull.closed_at, line))

    # Create sections
    sections = ["This release includes"]
    for label in sorted(lines):
        sections.append('\n' + label)
        lines[label].sort()  # sort by merge time
        sections.append('\n'.join(line for _, line in lines[label]))
    notes = '\n'.join(sections)

    print(notes)


if __name__ == '__main__':

    PREVIOUS_RELEASE = datetime.datetime(2020, 1, 17)
    LABELS = ('bug', 'fix for dataset')

    draft_notes_since(PREVIOUS_RELEASE, LABELS)
