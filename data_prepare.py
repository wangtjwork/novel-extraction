from FileProcess import FileProcess

f = FileProcess('data/001_to_kill_a_mockingbird.txt')
f.getMainCharacter(5)
f.filterFile('Jem', 'newfile')