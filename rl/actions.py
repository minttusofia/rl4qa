def action_space_for_id(actions_id):
    action_spaces = {
        'located_in_the_administrative_territorial_entity':
            [['where is'],
             ['municipality of'],
             ['located'],
             ['what kind of socks does', 'wear']],
        'occupation':
            [['employer of'],
             ['colleague of'],
             ['known for'],
             ['work as']],
        'all-small':
            [
                ["what is"],
                ["where is"],
                ["who is"],
                ["what does"],

                ["what is", "a part of"],
                ["what is", "an instance of"],
                ["who made"],

                ["what does", "work as"],
                ["where does", "work"],
                ["what does", "do"],

                ["where was", "born"],
                ["where is", "from"],

                ["who is the employer of"],
                ["who is related to"],
                ["who is friend of"],
                ["who does", "work with"],
                ["who likes"],
                ["who knows"],

                ["what is", "known for"],
                ["what type is"],

                ["how big is"],
                ["how old is"],

                ["what genre is"],

                ["when was", "born"],
                ["when did", "die"],

                ["what language does", "speak"],
                ["nationality of"],
                ["what language is", "in"],

                ["publisher of"],
                ["when was", "published"],
            ],

        'all':  # from baselines/template_list_70.json
            [
                ["what is"],
                ["where is"],
                ["who is"],
                ["what does"],

                ["what is", "a part of"],
                ["what is", "an instance of"],
                ["who made"],

                ["where is", "located"],
                ["municipality of"],
                ["what administrative territorial entity is", "located in"],
                ["where can", "be found"],
                ["where is", "situated"],

                ["what does", "work as"],
                ["where does", "work"],
                ["what company does", "work for"],
                ["what does", "do"],
                ["what is the profession of"],
                ["what is the occupation of"],
                ["what is", "by trade"],
                ["what is the job of"],

                ["where was", "born"],
                ["where is", "from"],
                ["what country is", "from"],
                ["hometown of"],
                ["what city is", "from"],

                ["what is the favourite polar bear of"],
                ["what kind of socks does", "wear"],
                ["how often does", "sneeze"],

                ["who is the employer of"],
                ["who is the mother of"],
                ["who is the father of"],
                ["who is the son of"],
                ["who is the daughter of"],
                ["who is related to"],
                ["who is friend of"],
                ["who is friends with"],
                ["who likes"],
                ["who knows"],

                ["what is", "known for"],
                ["what type is"],
                ["what kind is"],
                ["what class is"],

                ["how big is"],
                ["how old is"],
                ["how large is"],
                ["what is the area of"],

                ["what label does", "produce music under"],
                ["what label is", "signed under"],

                ["what genre is"],
                ["what is the genre of"],
                ["what type of movie is"],
                ["what type of artwork is"],
                ["what type of book is"],

                ["when was", "born"],
                ["when did", "die"],

                ["what language does", "speak"],
                ["native language of"],
                ["first language of"],
                ["nationality of"],
                ["what language is", "in"],

                ["publisher of"],
                ["who was", "published by"],

                ["what date was", "published"],
                ["when was", "published"],

                ["affiliated with"],
                ["who is the colleague of"],
                ["who does", "work with"],
                ["political leaning of"],
                ["what is", "'s political party"],

                ["what is", "part of"]
            ]

    }
    return action_spaces[actions_id]

