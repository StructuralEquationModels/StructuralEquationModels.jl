t = Template(;
           user="Maximilian-Stefan-Ernst",
           license="MIT",
           authors=["Maximilian Ernst", "Aaron Peikert"],
           dir="C:/Users/maxim/github/julia_sem",
           julia_version=v"1.2",
           plugins=[
               TravisCI(),
               Codecov(),
               GitHubPages(),
           ],
       )

generate(t, "sem")
