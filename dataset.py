from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

my_query_templates = [
"Is <sks> in this photo?"
"Is <sks> visible in this photo?",
"Is <sks> present in this photo?",
"Is <sks> captured in this photo?",
"Is <sks> included in this photo?",
"Is <sks> featured in this photo?",
"Is <sks> depicted in this photo?",
"Is <sks> shown in this photo?",
"Is <sks> part of this photo?",
"Is <sks> seen in this photo?",
"Is <sks> identifiable in this photo?",
]

class PersonalizedDataset_Two(Dataset):
    def __init__(
        self,
        data_root,
        sks_name,
        sks_name_2,
        tokenizer,
        set="train",
        placeholder_token="<sks>",
        center_crop=False,
        device="cuda",
        config=None,
        image_processor=None,
        json_path = './GPT4/training_data/',
        flip_p=0.5,
        train_lm_head=False,
        extreme_negative=False,
        recog_only=False,
        random_image=False,
        text_only=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.image_processor = image_processor
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.sks_name = sks_name
        self.sks_name_2 = sks_name_2
        self.questions = []
        self.images_path = []
        self.answers = []
        self.has_image = []

        # --- Load data from json files
        if recog_only:
            if extreme_negative:
                conversation_types = ['recognition_positive', 'recognition_negative-stuffed-animals']
            else:
                if text_only:
                    conversation_types = ['recognition_positive', 'recognition_negative-laion-mix', 'text-only-conversation']
                else:
                    conversation_types = ['recognition_positive', 'recognition_negative-laion-mix']
        else:
            if extreme_negative:
                conversation_types = ['recognition_positive', 'recognition_negative-stuffed-animals', 'conversation', 'text-only-conversation', 'complex_reasoning', 'detail_description']
            else:
                conversation_types = ['recognition_positive', 'recognition_negative-laion', 'conversation', 'text-only-conversation', 'complex_reasoning', 'detail_description']
        if random_image:
            conversation_types.extend(['recognition_negative-cc12m-mix'])

        for conversation_type in conversation_types:
            for sks in [sks_name, sks_name_2]:
                f = open(os.path.join(json_path, sks, f'{conversation_type}.json'))
                data = json.load(f)
                file_names = [x for x in data.keys()]
                for file_name in file_names:
                    questions = []
                    answers = []
                    for conv in data[file_name]:
                        # -- Filter out the conversation with only one question or one answer (Not full conversation)
                        if len(conv.keys()) ==2:
                            questions.append(conv['Human'])
                            answers.append(conv['AI'])
                    self.questions.extend(questions)
                    self.answers.extend(answers)
                    
                    #-- Same image for these conversations
                    # if ('laion' in file_name) or ('cc12m' in file_name):
                    #     file_name = file_name.replace('/nobackup/thao-data/code/my-llava/GPT4', './GPT4')
                    #     file_name = file_name.replace('./training_data', './GPT4/training_data')
                    #     file_name = file_name.replace('../', './')
                    #     file_name = file_name.replace('/YourLLaVA/', '/stuffed-animals/')
                    # else:
                    #     file_name = file_name.replace('/nobackup/thao-data/dataset/', './dataset/')
                    #     file_name = file_name.replace('../', './')
                    #     file_name = file_name.replace('/YourLLaVA/', '/stuffed-animals/')
                    
                    self.images_path.extend([file_name]*len(answers))
                    if conversation_type == 'text-only-conversation':
                        self.has_image.extend([False]*len(answers))
                    else:
                        self.has_image.extend([True]*len(answers))
                print(conversation_type, len(self.questions))
        
        if set == "train":
            self._length = len(self.questions)
        else:
            self._length = self.num_images
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        # self.templates = my_query_templates

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        # --- Center crop 
        # if self.center_crop:
        #     crop = min(img.shape[0], img.shape[1])
        #     (
        #         h,
        #         w,
        #     ) = (
        #         img.shape[0],
        #         img.shape[1],
        #     )
        #     img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        image_path = self.images_path[i]
        images = [Image.open(image_path).convert("RGB")]

        # --- Maybe flip the image...
        images = [self.flip_transform(image) for image in images]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.config
        )
        example["images"] = images_tensor
        # example['query'] = self.questions[i].replace('<sks>', f'<{self.sks_name}>')
        # example['query'] = example['query'].replace('<sks2>', f'<{self.sks_name_2}>')

        # example['answer'] = self.answers[i].replace('<sks>', f'<{self.sks_name}>')
        # example['answer'] = example['answer'].replace('<sks2>', f'<{self.sks_name_2}>')
        # print(example['query'])
        # print(example['answer'])
        # print(example['query'], example['answer'])
        example['has_image'] = self.has_image[i]
        example['image_sizes'] = image_sizes
        return example

class PersonalizedDataset_Mixture(Dataset):
    def __init__(
        self,
        data_root,
        sks_name,
        tokenizer,
        set="train",
        placeholder_token="<sks>",
        center_crop=False,
        device="cuda",
        config=None,
        image_processor=None,
        json_path = './training-data',
        flip_p=0.5,
        train_lm_head=False,
        extreme_negative=False,
        recog_only=False,
        random_image=False,
        text_only=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.image_processor = image_processor
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.sks_name = sks_name
        self.questions = []
        self.images_path = []
        self.answers = []
        self.has_image = []
        # --- Load data from json files
        if recog_only:
            if text_only:
                conversation_types = ['positive_recognition', 'negative_recognition-laion', 'text-only-conversation']
            else:
                conversation_types = ['positive_recognition', 'negative_recognition-laion']
        else:
            conversation_types = ['positive_recognition', 'negative_recognition-laion', 'conversation', 'text-only-conversation', 'complex_reasoning', 'detail_description']
        if random_image:
            conversation_types.extend(['negative_recognition-random-imgs'])
        # conversation_types = ['recognition_positive', 'recognition_negative-cc12m']
        # for conversation_type in ['negative_example-cc12m', 'conversation', 'text-only-conversation', 'complex_reasoning', 'detail_description']:
        # for conversation_type in ['recognition_positive', 'recognition_negative-cc12m']:
        # for conversation_type in ['detail_description']:
        for conversation_type in conversation_types:
            
            f = open(os.path.join(json_path, sks_name, f'{conversation_type}.json'))
            data = json.load(f)
            file_names = [x for x in data.keys()]
            for file_name in file_names:
                questions = []
                answers = []
                for conv in data[file_name]:
                    questions.append(conv['Human'])
                    answers.append(conv['AI'])
                self.questions.extend(questions)
                self.answers.extend(answers)
                
                self.images_path.extend([file_name]*len(answers))
                if conversation_type == 'text-only-conversation':
                    self.has_image.extend([False]*len(answers))
                else:
                    self.has_image.extend([True]*len(answers))
            print(conversation_type, len(self.questions))
        print('Total: ', len(self.questions), len(self.answers), len(self.images_path), len(self.has_image))
        if set == "train":
            self._length = len(self.questions)
        else:
            self._length = self.num_images
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        # self.templates = my_query_templates

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        # --- Center crop 
        # if self.center_crop:
        #     crop = min(img.shape[0], img.shape[1])
        #     (
        #         h,
        #         w,
        #     ) = (
        #         img.shape[0],
        #         img.shape[1],
        #     )
        #     img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        image_path = self.images_path[i]
        images = [Image.open(image_path).convert("RGB")]

        # --- Maybe flip the image...
        images = [self.flip_transform(image) for image in images]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.config
        )
        example["images"] = images_tensor
        example['query'] = self.questions[i]
        example['answer'] = self.answers[i]
        example['has_image'] = self.has_image[i]
        example['image_sizes'] = image_sizes
        return example

class PersonalizedDataset(Dataset):
    def __init__(
        self,
        data_root,
        sks_name,
        train_image_paths,
        tokenizer,
        size=512,
        repeats=1,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        center_crop=False,
        device="cuda",
        config=None,
        image_processor=None,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.image_processor = image_processor
        # self.learnable_property = learnable_property
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.image_paths = []
        for x in np.unique(train_image_paths):
            self.image_paths.append(x)

        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.sks_name = sks_name
        if set == "train":
            self._length = self.num_images * repeats
        else:
            self._length = self.num_images

        self.templates = my_query_templates
        # self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self.num_images]
        images = [Image.open(image_path).convert("RGB")]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.config
        )#.to(dtype=torch.float16)
        example["images"] = images_tensor
        example["query"] = f'Can you see <{self.sks_name}> in this photo? Answer the question using a single word or phrase.'

        # if 'laion' in image_path:
        if ('laion' in image_path) or ('random-imgs' in image_path):
            example["answer"] = 'No'
        else:
            example["answer"] = 'Yes'
        example['image_sizes'] = image_sizes
        example['has_image'] = True
        return example

if __name__ == "__main__":


    eval_model(args)
