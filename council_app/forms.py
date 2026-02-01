from django import forms
from .models import Query, ModelConfig, CreativeProject, ChurnIteration


class QueryForm(forms.ModelForm):
    """Form for submitting a new query to the council"""
    
    models = forms.ModelMultipleChoiceField(
        queryset=ModelConfig.objects.filter(is_active=True),
        widget=forms.CheckboxSelectMultiple,
        required=True,
        help_text="Select at least 2 models for the council"
    )
    
    class Meta:
        model = Query
        fields = ['prompt', 'council_mode']
        widgets = {
            'prompt': forms.Textarea(attrs={
                'rows': 4,
                'placeholder': 'Enter your question or prompt...',
                'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none'
            }),
            'council_mode': forms.Select(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent'
            }),
        }
        labels = {
            'prompt': 'Your Question',
            'council_mode': 'Voting Mode',
        }
    
    def clean_models(self):
        models = self.cleaned_data.get('models')
        if models and len(models) < 2:
            raise forms.ValidationError("Please select at least 2 models for the council.")
        return models
    
    def clean_prompt(self):
        prompt = self.cleaned_data.get('prompt', '').strip()
        if len(prompt) < 10:
            raise forms.ValidationError("Please enter a more detailed prompt (at least 10 characters).")
        return prompt


class ModelConfigForm(forms.ModelForm):
    """Form for configuring a model"""
    
    class Meta:
        model = ModelConfig
        fields = ['name', 'display_name', 'is_active', 'weight']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500',
                'placeholder': 'e.g., phi3:mini'
            }),
            'display_name': forms.TextInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500',
                'placeholder': 'e.g., Phi-3 Mini'
            }),
            'weight': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500',
                'step': '0.1',
                'min': '0.1',
                'max': '5.0'
            }),
        }


class AddModelForm(forms.Form):
    """Simple form for adding a new model by name"""
    
    model_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500',
            'placeholder': 'e.g., llama3:8b'
        }),
        help_text="Enter the Ollama model name"
    )
    
    def clean_model_name(self):
        name = self.cleaned_data.get('model_name', '').strip()
        if ModelConfig.objects.filter(name=name).exists():
            raise forms.ValidationError("This model is already configured.")
        return name


# =============================================================================
# CHURN MACHINE FORMS
# =============================================================================

class CreativeProjectForm(forms.ModelForm):
    """Form for creating a new creative writing project"""
    
    models = forms.ModelMultipleChoiceField(
        queryset=ModelConfig.objects.filter(is_active=True),
        widget=forms.CheckboxSelectMultiple,
        required=True,
        help_text="Select at least 2 models for the council"
    )
    
    class Meta:
        model = CreativeProject
        fields = ['title', 'description', 'content_type', 'processing_mode', 'default_churn_type']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent',
                'placeholder': 'Enter project title...'
            }),
            'description': forms.Textarea(attrs={
                'rows': 3,
                'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none',
                'placeholder': 'Describe your project (optional)...'
            }),
            'content_type': forms.Select(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent'
            }),
            'processing_mode': forms.Select(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent'
            }),
            'default_churn_type': forms.Select(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent'
            }),
        }
        labels = {
            'title': 'Project Title',
            'description': 'Description',
            'content_type': 'Content Type',
            'processing_mode': 'Processing Mode',
            'default_churn_type': 'Default Churn Type',
        }
    
    def clean_title(self):
        title = self.cleaned_data.get('title', '').strip()
        if len(title) < 3:
            raise forms.ValidationError("Please enter a longer title (at least 3 characters).")
        return title
    
    def clean_models(self):
        models = self.cleaned_data.get('models')
        if models and len(models) < 2:
            raise forms.ValidationError("Please select at least 2 models for the council.")
        return models


class SubmitContentForm(forms.Form):
    """Form for submitting content for churning"""
    
    content = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 15,
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent font-mono text-sm',
            'placeholder': 'Paste or type your writing here...'
        }),
        label='Your Writing',
        help_text='Enter the text you want the council to review'
    )
    
    churn_type = forms.ChoiceField(
        choices=CreativeProject.ChurnType.choices,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent'
        }),
        label='Churn Type',
        help_text='How should the council process this content?'
    )
    
    context = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent',
            'placeholder': 'e.g., "Horror short story" or "Business blog post"'
        }),
        label='Context/Genre',
        help_text='Optional context to help guide the feedback'
    )
    
    models = forms.ModelMultipleChoiceField(
        queryset=ModelConfig.objects.filter(is_active=True),
        widget=forms.CheckboxSelectMultiple,
        required=True,
        help_text="Select at least 2 models for the council"
    )
    
    def clean_content(self):
        content = self.cleaned_data.get('content', '').strip()
        if len(content) < 50:
            raise forms.ValidationError("Please enter more content (at least 50 characters).")
        return content
    
    def clean_models(self):
        models = self.cleaned_data.get('models')
        if models and len(models) < 2:
            raise forms.ValidationError("Please select at least 2 models for the council.")
        return models


class BranchForm(forms.Form):
    """Form for creating a branch from an iteration"""
    
    branch_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent',
            'placeholder': 'e.g., "Darker tone" or "Alternative ending"'
        }),
        label='Branch Name',
        help_text='Give this branch a descriptive name'
    )
    
    direction = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'rows': 3,
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none',
            'placeholder': 'Describe the direction you want to explore (optional)...'
        }),
        label='Exploration Direction',
        help_text='Optional guidance for how this branch should differ'
    )
    
    use_suggested_content = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded'
        }),
        label='Start from suggested content',
        help_text='Use the council\'s suggested revision as the starting point'
    )
    
    def clean_branch_name(self):
        name = self.cleaned_data.get('branch_name', '').strip()
        if len(name) < 2:
            raise forms.ValidationError("Please enter a longer branch name.")
        return name


class BatchChurnForm(forms.Form):
    """Form for running multiple churn iterations automatically"""
    
    num_iterations = forms.IntegerField(
        min_value=1,
        max_value=10,
        initial=3,
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent',
            'min': '1',
            'max': '10'
        }),
        label='Number of Iterations',
        help_text='How many iterations to run (1-10)'
    )
    
    auto_apply = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded'
        }),
        label='Auto-apply suggestions',
        help_text='Automatically apply each iteration\'s suggestions before the next'
    )
    
    churn_type = forms.ChoiceField(
        choices=CreativeProject.ChurnType.choices,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent'
        }),
        label='Churn Type',
        help_text='Type of processing for all iterations'
    )
    
    models = forms.ModelMultipleChoiceField(
        queryset=ModelConfig.objects.filter(is_active=True),
        widget=forms.CheckboxSelectMultiple,
        required=True,
        help_text="Select at least 2 models for the council"
    )
    
    def clean_models(self):
        models = self.cleaned_data.get('models')
        if models and len(models) < 2:
            raise forms.ValidationError("Please select at least 2 models for the council.")
        return models


class TriggerChurnForm(forms.Form):
    """Form for triggering a single churn iteration"""
    
    churn_type = forms.ChoiceField(
        choices=CreativeProject.ChurnType.choices,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent'
        }),
        label='Churn Type'
    )
    
    use_suggested_content = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded'
        }),
        label='Use suggested content',
        help_text='Start from the previous iteration\'s suggested revision'
    )
    
    models = forms.ModelMultipleChoiceField(
        queryset=ModelConfig.objects.filter(is_active=True),
        widget=forms.CheckboxSelectMultiple,
        required=True,
        help_text="Select at least 2 models for the council"
    )
    
    def clean_models(self):
        models = self.cleaned_data.get('models')
        if models and len(models) < 2:
            raise forms.ValidationError("Please select at least 2 models for the council.")
        return models
