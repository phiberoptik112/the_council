"""
Management command to seed default models for the council.
"""
from django.core.management.base import BaseCommand
from council_app.models import ModelConfig


DEFAULT_MODELS = [
    {'name': 'phi3:mini', 'display_name': 'Phi-3 Mini'},
    {'name': 'tinyllama', 'display_name': 'TinyLlama'},
    {'name': 'qwen2.5:0.5b', 'display_name': 'Qwen 2.5 (0.5B)'},
    {'name': 'gemma2:2b', 'display_name': 'Gemma 2 (2B)'},
    {'name': 'llama3.2:1b', 'display_name': 'Llama 3.2 (1B)'},
]


class Command(BaseCommand):
    help = 'Seed the database with default Ollama models'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing models before seeding',
        )

    def handle(self, *args, **options):
        if options['clear']:
            ModelConfig.objects.all().delete()
            self.stdout.write(self.style.WARNING('Cleared all existing models'))

        created_count = 0
        for model_data in DEFAULT_MODELS:
            model, created = ModelConfig.objects.get_or_create(
                name=model_data['name'],
                defaults={'display_name': model_data['display_name']}
            )
            if created:
                created_count += 1
                self.stdout.write(f"  Created: {model.name}")
            else:
                self.stdout.write(f"  Exists: {model.name}")

        self.stdout.write(self.style.SUCCESS(f'\nSeeded {created_count} new models'))
        self.stdout.write(f'Total models: {ModelConfig.objects.count()}')
        self.stdout.write('\nNote: Make sure these models are pulled in Ollama:')
        for model_data in DEFAULT_MODELS:
            self.stdout.write(f"  ollama pull {model_data['name']}")
