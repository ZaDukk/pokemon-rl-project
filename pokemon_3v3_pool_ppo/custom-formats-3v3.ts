// Copy to pokemon-showdown/config/custom-formats.ts and RESTART Showdown.
// This includes your earlier RL 1v1 and RL 2v2 formats, plus the new 3v3.
// Back up your installed custom-formats.ts first if you have other formats.
export const Formats: import('../sim/dex-formats').FormatList = [
  {
    name: '[Gen 9] RL 1v1',
    mod: 'gen9', gameType: 'singles', searchShow: false, rated: false,
    ruleset: ['Obtainable', 'Min Team Size = 1', 'Max Team Size = 1',
              'HP Percentage Mod', 'Cancel Mod', 'Endless Battle Clause', 'Terastal Clause'],
  },
  {
    name: '[Gen 9] RL 2v2 Singles',
    mod: 'gen9', gameType: 'singles', searchShow: false, rated: false,
    ruleset: ['Obtainable', 'Min Team Size = 2', 'Max Team Size = 2',
              'HP Percentage Mod', 'Cancel Mod', 'Endless Battle Clause', 'Terastal Clause'],
  },
  {
    name: '[Gen 9] RL 3v3 Singles',
    mod: 'gen9', gameType: 'singles', searchShow: false, rated: false,
    ruleset: ['Obtainable', 'Min Team Size = 3', 'Max Team Size = 3',
              'Species Clause', 'HP Percentage Mod', 'Cancel Mod',
              'Endless Battle Clause', 'Terastal Clause'],
  },
];
