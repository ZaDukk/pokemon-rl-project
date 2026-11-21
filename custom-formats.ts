// Replace pokemon-showdown/config/custom-formats.ts with this, and RESTART Showdown.
// Includes your original 1v1 format as well as the new two-Pokemon SINGLE battle.
export const Formats: import('../sim/dex-formats').FormatList = [
  {
    name: '[Gen 9] RL 1v1',
    mod: 'gen9',
    gameType: 'singles',
    searchShow: false,
    rated: false,
    ruleset: [
      'Obtainable',
      'Min Team Size = 1',
      'Max Team Size = 1',
      'HP Percentage Mod',
      'Cancel Mod',
      'Endless Battle Clause',
      'Terastal Clause',
    ],
  },
  {
    name: '[Gen 9] RL 2v2 Singles',
    mod: 'gen9',
    gameType: 'singles', // one active Pokemon per side; switching is enabled
    searchShow: false,
    rated: false,
    ruleset: [
      'Obtainable',
      'Min Team Size = 2',
      'Max Team Size = 2',
      'HP Percentage Mod',
      'Cancel Mod',
      'Endless Battle Clause',
      'Terastal Clause',
      // No Team Preview; first team member automatically leads.
    ],
  },
];
