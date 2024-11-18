import os
import streamlit as st
import torch.cuda
import samplerate
from st_session_management import set_session_state_key
from utils import generate_string_bracket_combinations, save_wav_audio
from models import get_pipe, set_torch_seed, run_pipe


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def welcome_intro():
    st.title("📢 Audio Generator 📢")
    st.markdown("""
    Welcome to the AI Audio generator.


    Use the options below to generate one or more pieces.
    """)


def set_prompt():
    columns = st.columns(2)
    with columns[0]:
        with st.form("Prompt", clear_on_submit=True, border=False):
            positive_prompt = st.text_input("**Positive prompt**",
                                            key='text_input',
                                            label_visibility="visible")
            prompts = generate_string_bracket_combinations(positive_prompt)
            submit_prompt = st.form_submit_button("Submit", type='primary')
            if st.session_state.get('positive_prompt') != prompts and prompts != [""] and submit_prompt:
                set_session_state_key('positive_prompt', prompts)
    with columns[1]:
        with st.form("Negative Prompt", clear_on_submit=True, border=False):
            negative_prompt = st.text_input("**Negative prompt**",
                                            key='neg_text_input',
                                            label_visibility="visible")
            submit_negprompt = st.form_submit_button("Submit", type='primary')
            if (
                    st.session_state.get('negative_prompt') != negative_prompt and
                    negative_prompt != "" and submit_negprompt
            ):
                set_session_state_key('negative_prompt', negative_prompt)


def print_current_prompts():
    with st.expander('**Current**', expanded=False):
        st.markdown('Positive Prompts: ' + (
            ", ".join([f"*{t}*" for t in st.session_state['positive_prompt']])
            if st.session_state['positive_prompt'] else "*None*"

        ))
        st.markdown(f'Negative Prompt: *{st.session_state["negative_prompt"]}*')


def set_model_parameters():
    columns = st.columns(5)
    with columns[0]:
        pretrained_model_name_or_path = st.selectbox("Model Name",
                                                     options=['audioldm2', 'audioldm2-large',
                                                              'stable-audio-open-1.0'])
        if 'ldm2' in pretrained_model_name_or_path:
            pretrained_model_name_or_path = 'cvssp/' + pretrained_model_name_or_path
        else:
            pretrained_model_name_or_path = 'stabilityai/' + pretrained_model_name_or_path
        specific_folder = st.text_input('Set specific folder', None)
    with columns[1]:
        device = st.selectbox('Device', options=(["cuda"] if torch.cuda.is_available() else []) + ['cpu'],
                              index=0)
        audio_length_in_s = st.number_input('Audio length [s]', min_value=1., value=5., step=0.5, format='%.2f')

    with columns[2]:
        num_waveforms_per_prompt = st.number_input('num waveforms per prompt', min_value=1, value=3, format='%d')
        num_outputs_per_prompt = st.number_input('num outputs per prompt', min_value=1, value=4, format='%d')

    with columns[3]:
        num_inference_steps = st.number_input('num inference steps', min_value=10,
                                              value=25 if 'ldm2' in pretrained_model_name_or_path else 100,
                                              max_value=500, step=5,
                                              format='%d')
        guidance_scale = st.number_input('Guidance Scale', min_value=1., max_value=10.,
                                         value=3.5 if 'ldm2' in pretrained_model_name_or_path else 7.5, step=0.5,
                                         format='%.2f')
    with columns[4]:
        ratio = st.selectbox("Ratio [kHz]", options=[16, 48])
        final_ratio = st.number_input("End Ratio [kHz]", value=44.1, max_value=48., format='%.2f')

    return (pretrained_model_name_or_path, final_ratio, audio_length_in_s, num_waveforms_per_prompt,
            num_outputs_per_prompt, num_inference_steps, guidance_scale, device, ratio, specific_folder)


def get_pipeline():
    with st.form('Pipeline'):
        (pretrained_model_name_or_path, final_ratio, audio_length_in_s, num_waveforms_per_prompt,
         num_outputs_per_prompt, num_inference_steps, guidance_scale, device, ratio,
         specific_folder) = set_model_parameters()
        submit = st.form_submit_button("Submit pipeline parameters")
        if submit:
            st.session_state["pretrained_model_name_or_path"] = pretrained_model_name_or_path
            st.session_state['final_ratio'] = final_ratio
            st.session_state['audio_length_in_s'] = audio_length_in_s
            st.session_state['num_waveforms_per_prompt'] = num_waveforms_per_prompt
            st.session_state['num_outputs_per_prompt'] = num_outputs_per_prompt
            st.session_state['num_inference_steps'] = num_inference_steps
            st.session_state['guidance_scale'] = guidance_scale
            st.session_state["device"] = device
            st.session_state["ratio"] = ratio
            if specific_folder:
                st.session_state['save_folder'] = os.path.join(st.session_state['reporting'], specific_folder)
                os.makedirs(st.session_state['save_folder'], exist_ok=True)
            else:
                st.session_state['save_folder'] = st.session_state['reporting']
            st.session_state['pipeline'] = get_pipe(st.session_state['pretrained_model_name_or_path'],
                                                    st.session_state['device'])


def run_pipeline():
    if st.button('Generate', type='primary', disabled=any([not st.session_state['pipeline'],
                                                           not st.session_state['positive_prompt']])):
        for i in range(st.session_state['num_outputs_per_prompt']):
            for prompt in st.session_state['positive_prompt']:
                audio = run_pipe(pipe=st.session_state['pipeline'],
                                 prompt=prompt,
                                 negative_prompt=st.session_state['negative_prompt'],
                                 audio_length_in_s=st.session_state['audio_length_in_s'],
                                 random_generator=set_torch_seed(i, st.session_state['device']),
                                 guidance_scale=st.session_state['guidance_scale'],
                                 num_inference_steps=st.session_state['num_inference_steps'],
                                 num_waveforms_per_prompt=st.session_state['num_waveforms_per_prompt'],
                                 initial_audio_sampling_rate=st.session_state['final_ratio'])
                audio = (samplerate.resample(audio[0].detach().cpu().numpy(),
                                             st.session_state['final_ratio'] / st.session_state['ratio'],
                                             'sinc_best')
                         if st.session_state['final_ratio'] != st.session_state['ratio'] else
                         audio[0])
                save_wav_audio(audio, st.session_state['save_folder'],
                               prompt.replace(' ', '_'),
                               rate=int(st.session_state['final_ratio'] * 1000))
        return True
