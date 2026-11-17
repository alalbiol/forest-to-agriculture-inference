"""Set the GitHub repository description, website, and topics after gh auth login."""
import os
import subprocess

REPOSITORY = 'alalbiol/forest-to-agriculture-inference'
DESCRIPTION = ('Score Sentinel-1/2 chip pairs for forest-to-agriculture conversion with a frozen '
               'TerraFM backbone and a temporal land-use rule; pretrained weights on EOTDL.')
HOMEPAGE = 'https://www.eotdl.com/models/forest-to-agriculture-inference'
TOPICS = ['deforestation', 'land-use-change', 'sentinel-2', 'sentinel-1', 'foundation-models',
          'terrafm', 'eudr', 'pytorch', 'remote-sensing', 'earth-engine']


def main():
    env = {k: v for k, v in os.environ.items() if k not in ('GH_TOKEN', 'GITHUB_TOKEN', 'GH_DEBUG')}
    subprocess.run(['gh', 'auth', 'status', '--hostname', 'github.com'], env=env, check=True)
    command = ['gh', 'repo', 'edit', REPOSITORY, '--description', DESCRIPTION, '--homepage', HOMEPAGE]
    for topic in TOPICS:
        command.extend(['--add-topic', topic])
    subprocess.run(command, env=env, check=True)
    print(f'Updated About fields for {REPOSITORY}.')


if __name__ == '__main__':
    main()
